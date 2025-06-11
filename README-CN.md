[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 Seed-VC 变声节点

非常快速地将 `语音` 或 `歌声` 变成另一个人的声音, 不改变内容节奏.

## 📣 更新

[2025-06-05]⚒️: 发布 v1.0.0.

## 使用

![image](https://github.com/billwuhao/ComfyUI_Seed-VC/blob/main/images/2025-06-05_01-43-33.png)

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Seed-VC.git
cd ComfyUI_Seed-VC
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

手动下载放到 `ComfyUI\models\TTS\Seed-VC` 路径下的模型:

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

手动下载 **文件夹** 放到 `ComfyUI\models\TTS` 路径下:

- [bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/tree/main) 只需要 `bigvgan_generator.pt` 文件和参数文件.

```
    .
    │  bigvgan_generator.pt
    │  config.json
```

- [bigvgan_v2_44khz_128band_512x](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/tree/main) 只需要 `bigvgan_generator.pt` 文件和参数文件.

```
    .
    │  bigvgan_generator.pt
    │  config.json
```

- [whisper-small](https://huggingface.co/openai/whisper-small/tree/main) 只需要 `model.safetensors` 文件和 `.json`, 参数小文件.

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

## 鸣谢

[seed-vc](https://github.com/Plachtaa/seed-vc)
