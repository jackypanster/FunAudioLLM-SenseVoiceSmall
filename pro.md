实现一个语音转录文本的需求

1. 选择的模型：https://github.com/FunAudioLLM/SenseVoice

2. huggingface 地址://huggingface.co/FunAudioLLM/SenseVoiceSmall， 模型已经下载到本地目录：/home/llm/model/iic/SenseVoiceSmall

3. 用fatsapi.py 提供一个 restful api 给用户使用，譬如这样：curl -X POST "http://127.0.0.1:8000/asr_pure" -F "file=@test_audio.wav"

4. 希望用 uv（https://docs.astral.sh/uv/guides/install-python/） 管理后端python开发环境

5. 系统是ubuntu 24.04, 有4个 Nvidia 2080Ti 显卡，每个显卡22G，一共88G显存。物理内存 512G，磁盘 2T SSD，56核 CPU。

6. 更多系统环境：NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4 

7. which python
/home/llm/miniconda3/bin/python

8. which pip
/home/llm/miniconda3/bin/pip

