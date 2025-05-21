# 语音转录文本服务需求文档

## 1. 需求概述

实现一个将语音实时转录为文本的服务。

## 2. 模型选型

*   **模型名称**: SenseVoice
*   **项目地址**: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
*   **具体模型**: SenseVoiceSmall
*   **Hugging Face 地址**: [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
*   **本地模型路径**: `/home/llm/model/iic/SenseVoiceSmall` (已下载)

## 3. API 接口设计

*   **框架**: FastAPI
*   **端点**: `POST /asr_pure`
*   **请求示例**:
    ```bash
    curl -X POST "http://127.0.0.1:8000/asr_pure" -F "file=@test_audio.wav"
    ```

## 4. 技术栈

*   **Python 环境管理**: `uv` ([Astral uv Documentation](https://docs.astral.sh/uv/guides/install-python/))

## 5. 硬件环境

*   **操作系统**: Ubuntu 24.04
*   **GPU**: 4 x Nvidia 2080Ti
    *   单卡显存: 22GB
    *   总显存: 88GB
*   **物理内存 (RAM)**: 512GB
*   **磁盘**: 2TB SSD
*   **CPU**: 56 核

## 6. 软件环境与驱动

*   **NVIDIA 驱动版本**: `550.144.03`
    *   `NVIDIA-SMI 550.144.03`
*   **CUDA 版本**: `12.4`
*   **Python 解释器路径**: `/home/llm/miniconda3/bin/python`
*   **pip 工具路径**: `/home/llm/miniconda3/bin/pip`

