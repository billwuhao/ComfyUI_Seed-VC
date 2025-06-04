import torch
import torchaudio
import librosa
import numpy as np
import yaml
from transformers import AutoFeatureExtractor, WhisperModel
import folder_paths
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.audio import mel_spectrogram
from modules.rmvpe import RMVPE

cache_dir = folder_paths.get_temp_directory()
models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS", "Seed-VC")


class SeedVCWrapper:
    def __init__(self, device=None):
        """
        Initialize the Seed-VC wrapper with all necessary models and configurations.
        
        Args:
            device: torch device to use. If None, will be automatically determined.
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # Load base model and configuration
        self._load_base_model()
        
        # Load F0 conditioned model
        self._load_f0_model()
        
        # Load additional modules
        self._load_additional_modules()
        
        # Set streaming parameters
        self.overlap_frame_len = 16
        self.bitrate = "320k"
        
    def _load_base_model(self):
        """Load the base DiT model for voice conversion."""
        dit_checkpoint_path = os.path.join(model_path, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth")
        dit_config_path = os.path.join(current_dir, "configs", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        
        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        self.model = build_model(model_params, stage='DiT')
        self.hop_length = config['preprocess_params']['spect_params']['hop_length']
        self.sr = config['preprocess_params']['sr']
        
        # Load checkpoints
        self.model, _, _, _ = load_checkpoint(
            self.model, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Set up mel spectrogram function
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        # Load whisper model
        whisper_name = os.path.join(models_dir, "TTS", "whisper-small")
        self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
    def _load_f0_model(self):
        """Load the F0 conditioned model for voice conversion."""
        dit_checkpoint_path = os.path.join(model_path, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth")
        dit_config_path = os.path.join(current_dir, "configs", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        
        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        self.model_f0 = build_model(model_params, stage='DiT')
        self.hop_length_f0 = config['preprocess_params']['spect_params']['hop_length']
        self.sr_f0 = config['preprocess_params']['sr']
        
        # Load checkpoints
        self.model_f0, _, _, _ = load_checkpoint(
            self.model_f0, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model_f0:
            self.model_f0[key].eval()
            self.model_f0[key].to(self.device)
        self.model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Set up mel spectrogram function for F0 model
        mel_fn_args_f0 = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr_f0,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)
        
    def _load_additional_modules(self):
        """Load additional modules like CAMPPlus, BigVGAN, and RMVPE."""
        # Load CAMPPlus
        campplus_ckpt_path = os.path.join(model_path, "campplus_cn_common.bin")
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)
        
        # Load BigVGAN models
        bigvgen_ckpt_path = os.path.join(models_dir, "TTS", "bigvgan_v2_22khz_80band_256x")
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgen_ckpt_path, use_cuda_kernel=False)
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval().to(self.device)
        
        bigvgen_ckpt_path_44k = os.path.join(models_dir, "TTS", "bigvgan_v2_44khz_128band_512x")
        self.bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained(bigvgen_ckpt_path_44k, use_cuda_kernel=False)
        self.bigvgan_44k_model.remove_weight_norm()
        self.bigvgan_44k_model = self.bigvgan_44k_model.eval().to(self.device)
        
        # Load RMVPE for F0 extraction
        rmvpe_model_path = os.path.join(model_path, "rmvpe.pt")
        self.rmvpe = RMVPE(rmvpe_model_path, is_half=False, device=self.device)
        
    @staticmethod
    def adjust_f0_semitones(f0_sequence, n_semitones):
        """Adjust F0 values by a number of semitones."""
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor
    
    @staticmethod
    def crossfade(chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2
    
    def _stream_wave_chunks(self, vc_wave, processed_frames, vc_target, overlap_wave_len, 
                           generated_wave_chunks, previous_chunk, is_last_chunk, sr):
        """
        Helper method to handle streaming wave chunks.
        
        Args:
            vc_wave: The current wave chunk
            processed_frames: Number of frames processed so far
            vc_target: The target mel spectrogram
            overlap_wave_len: Length of overlap between chunks
            generated_wave_chunks: List of generated wave chunks
            previous_chunk: Previous wave chunk for crossfading
            is_last_chunk: Whether this is the last chunk
            sr: Sample rate
            
        Returns:
            Tuple of (processed_frames, previous_chunk, should_break, mp3_bytes, full_audio)
            where should_break indicates if processing should stop
            mp3_bytes is the MP3 bytes if streaming, None otherwise
            full_audio is the full audio if this is the last chunk, None otherwise
        """
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                
                return processed_frames, previous_chunk, True, np.concatenate(generated_wave_chunks)
            
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
        elif is_last_chunk:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
            return processed_frames, previous_chunk, True, np.concatenate(generated_wave_chunks)
            
        else:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len
            
        return processed_frames, previous_chunk, False, None

    def _process_whisper_features(self, audio_16k, is_source=True):
        """Process audio through Whisper model to extract features."""
        if audio_16k.size(-1) <= 16000 * 30:
            # If audio is short enough, process in one go
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, :audio_16k.size(-1) // 320 + 1]
        else:
            # Process long audio in chunks
            overlapping_time = 5  # 5 seconds
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = audio_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer, 
                        audio_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features, attention_mask=inputs.attention_mask
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, :chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            features = torch.cat(features_list, dim=1)
        
        return features


    @torch.no_grad()
    def convert_voice(self, source, target, diffusion_steps=10, length_adjust=1.0,
                     inference_cfg_rate=0.7, f0_condition=False, auto_f0_adjust=True, 
                     pitch_shift=0):
        """
        Convert both timbre and voice from source to target.
        
        Args:
            source: Path to source audio file
            target: Path to target audio file
            diffusion_steps: Number of diffusion steps (default: 10)
            length_adjust: Length adjustment factor (default: 1.0)
            inference_cfg_rate: Inference CFG rate (default: 0.7)
            f0_condition: Whether to use F0 conditioning (default: False)
            auto_f0_adjust: Whether to automatically adjust F0 (default: True)
            pitch_shift: Pitch shift in semitones (default: 0)
            
        Returns:
            the full audio as a numpy array
        """
        # Select appropriate models based on F0 condition
        inference_module = self.model if not f0_condition else self.model_f0
        mel_fn = self.to_mel if not f0_condition else self.to_mel_f0
        bigvgan_fn = self.bigvgan_model if not f0_condition else self.bigvgan_44k_model
        sr = 22050 if not f0_condition else 44100
        hop_length = 256 if not f0_condition else 512
        max_context_window = sr // hop_length * 30
        overlap_wave_len = self.overlap_frame_len * hop_length
        
        # Load audio
        source_audio = librosa.load(source, sr=sr)[0]
        ref_audio = librosa.load(target, sr=sr)[0]
        
        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(self.device)
        
        # Resample to 16kHz for feature extraction
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        
        # Extract Whisper features
        S_alt = self._process_whisper_features(converted_waves_16k, is_source=True)
        S_ori = self._process_whisper_features(ref_waves_16k, is_source=False)
        
        # Compute mel spectrograms
        mel = mel_fn(source_audio.to(self.device).float())
        mel2 = mel_fn(ref_audio.to(self.device).float())
        
        # Set target lengths
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        
        # Compute style features
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))
        
        # Process F0 if needed
        if f0_condition:
            F0_ori = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
            F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)
            
            if self.device == "mps":
                F0_ori = torch.from_numpy(F0_ori).float().to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).float().to(self.device)[None]
            else:
                F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]
            
            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]
            
            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)
            
            # Shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if pitch_shift != 0:
                shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
        else:
            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None
        
        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
        
        # Process in chunks for streaming
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None
        
        # Generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                # Voice Conversion
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, mel2.size(-1):]
            
            vc_wave = bigvgan_fn(vc_target.float())[0]
            
            processed_frames, previous_chunk, should_break, full_audio = self._stream_wave_chunks(
                vc_wave, processed_frames, vc_target, overlap_wave_len, 
                generated_wave_chunks, previous_chunk, is_last_chunk, sr
            )
                
            if should_break:
                return full_audio, sr
            
        return np.concatenate(generated_wave_chunks), sr


from typing import Optional
import tempfile

def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


SEEDVC = None
class SeedVCRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",), 
                "ref_audio": ("AUDIO",), 
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}), 
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "inference_cfg_rate": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}), 
                "f0_condition": ("BOOLEAN", {"default": False, "tooltip": "Must set to true for singing voice conversion / 歌声转换时必须勾选"}), 
                "auto_f0_adjust": ("BOOLEAN", {"default": True}), 
                "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": True}),
                },
        }

    CATEGORY = "🎤MW/MW-Seed-VC"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    
    def run(self, source_audio, ref_audio, steps, speed, inference_cfg_rate, f0_condition, auto_f0_adjust, pitch_shift, unload_model):
        global SEEDVC
        if SEEDVC is None:
            SEEDVC = SeedVCWrapper()
            
        source_audio = cache_audio_tensor(cache_dir, source_audio["waveform"].squeeze(0), source_audio["sample_rate"])
        ref_audio = cache_audio_tensor(cache_dir, ref_audio["waveform"].squeeze(0), ref_audio["sample_rate"])
        audio, sr = SEEDVC.convert_voice(
            source=source_audio, 
            target=ref_audio, 
            diffusion_steps=steps, 
            length_adjust=speed, 
            inference_cfg_rate=inference_cfg_rate, 
            f0_condition=f0_condition, 
            auto_f0_adjust=auto_f0_adjust, 
            pitch_shift=pitch_shift, 
        )

        if unload_model:
            SEEDVC = None
            torch.cuda.empty_cache()

        return ({"waveform": torch.from_numpy(audio).unsqueeze(0).unsqueeze(0), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "SeedVCRun": SeedVCRun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVCRun": "Seed Voice Conversion"
}