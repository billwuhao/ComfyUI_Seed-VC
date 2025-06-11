[中文](README-CN.md) | [English](README.md) 

# ComfyUI's Seed-VC Voice Conversion Node

Very quickly convert `speech` or `singing` into another person's voice, without changing the content rhythm.

## 📣 Updates

[2025-06-05]⚒️: Released v1.0.0.

## Usage

![image](https://github.com/billwuhao/ComfyUI_Seed-VC/blob/main/images/2025-06-05_01-43-33.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Seed-VC.git
cd ComfyUI_Seed-VC
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

Manually download the models to the `ComfyUI\models\TTS\Seed-VC` path:

- [DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)
- [DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth)
- [campplus_cn_common.bin](https://huggingface.co/funasr/campplus/blob/main/campplus_cn_common.bin)
- [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

```
.
    campplus_cn_common.bin
    DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth
    DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
    rmvpe.pt
```

Manually download the folders to the `ComfyUI\models\TTS` path:

- [bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/main) Only the `bigvgan_generator.pt` file and parameter file are needed.

```
    .
    │  bigvgan_generator.pt
    │  config.json
```

- [bigvgan_v2_44khz_128band_512x](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/tree/main) Only the `bigvgan_generator.pt` file and parameter file are needed.

```
    .
    │  bigvgan_generator.pt
    │  config.json
```

- [whisper-small](https://huggingface.co/openai/whisper-small/tree/main) Only the `model.safetensors` file and `.json`, parameter files are needed.

```
.
    added_tokens.json
    config.json
    generation_config.json
    merges.txt
    model.safetensors
    normalizer.json
    preprocessor_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

## Acknowledgements

[seed-vc](https://github.com/Plachtaa/seed-vc)