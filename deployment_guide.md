 **部署文档**

**1. 引言**

本文档提供了在 Ubuntu 24.04 服务器上部署 FunAudioLLM/SenseVoiceSmall 语音转录服务的详细步骤。假设服务器已具备文档 `pro.md` 和 `architecture.md` 中描述的基础环境 (NVIDIA 驱动, CUDA)。

**2. 环境准备**

**2.1. 系统要求**

*   操作系统: Ubuntu 24.04
*   Python: `/home/llm/miniconda3/bin/python` (将通过 `uv` 管理项目特定环境)
*   NVIDIA 驱动: 550.144.03 或兼容版本
*   CUDA 版本: 12.4 或兼容版本
*   硬件: 至少一个 NVIDIA GPU (推荐 2080Ti 或更高)，足够的 CPU、内存和磁盘空间。

**2.2. 安装 `uv`**

如果尚未安装 `uv`，请参照官方文档安装：[https://docs.astral.sh/uv/guides/install/](https://docs.astral.sh/uv/guides/install/)

通常可以使用以下命令：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
安装完成后，验证安装：
```bash
~/.cargo/bin/uv --version 
# 或者按照安装脚本提示的路径执行 uv
```
建议将其路径添加到 `PATH` 环境变量中，例如在 `.bashrc` 或 `.zshrc` 中添加：
`export PATH="$HOME/.cargo/bin:$PATH"` (路径可能因安装方式而异)

**2.3. 项目代码获取**

将项目代码克隆或复制到服务器的目标部署目录，例如 `/srv/FunAudioLLM-SenseVoiceSmall`。

```bash
# 示例：如果使用 git 克隆
# git clone <your-repo-url> /srv/FunAudioLLM-SenseVoiceSmall
cd /srv/FunAudioLLM-SenseVoiceSmall
```

**3. 应用程序部署**

**3.1. 创建虚拟环境**

在项目根目录下，使用 `uv` 创建一个新的虚拟环境。建议使用与项目兼容的 Python 版本。

```bash
# 假设 /home/llm/miniconda3/bin/python 是 Python 3.9+
# 如果 uv 找不到该 python，可以直接指定路径
# uv venv --python /home/llm/miniconda3/bin/python .venv

uv venv .venv # uv 会尝试自动发现合适的 python 版本
```

激活虚拟环境：
```bash
source .venv/bin/activate
```

**3.2. 安装依赖**

项目依赖项在 `pyproject.toml` (推荐) 或 `requirements.txt` 文件中定义。

*   **如果使用 `pyproject.toml`**: (假设项目结构中包含此文件)
    ```bash
    uv pip install .
    ```
*   **如果使用 `requirements.txt`**: (根据 `architecture.md` 和 `detailed_design.md` 创建此文件)
    一个示例 `requirements.txt` 可能如下：
    ```txt
    fastapi
    uvicorn[standard] # standard 包含一些常用依赖
    python-multipart
    torch
    torchaudio
    transformers # 如果 SenseVoiceSmall 通过 transformers 加载
    # 其他 SenseVoiceSmall 模型可能需要的特定依赖
    # 例如：sentencepiece, pyyaml 等
    ```
    安装命令：
    ```bash
    uv pip install -r requirements.txt
    ```

**3.3. 模型文件**

确保 SenseVoiceSmall 模型文件已按 `architecture.md` 中所述放置在 `/home/llm/model/iic/SenseVoiceSmall`。
如果服务需要读取此目录，请确保运行服务的用户具有相应的读取权限。

**3.4. 配置环境变量 (可选但推荐)**

应用程序可能需要一些配置，可以通过环境变量传递。例如：

*   `MODEL_PATH`: SenseVoiceSmall 模型的路径 (默认为 `/home/llm/model/iic/SenseVoiceSmall`)
*   `DEVICE`: 使用的 GPU 设备 (例如 `cuda:0`, `cuda:1`)
*   `LOG_LEVEL`: 日志级别 (例如 `INFO`, `DEBUG`)
*   `MAX_UPLOAD_SIZE_MB`: 最大允许上传文件大小 (MB)
*   `ALLOWED_AUDIO_TYPES`: 允许的音频 MIME 类型 (逗号分隔, 例如 `audio/wav,audio/mpeg`)

可以在启动脚本中设置这些变量，或使用 `.env` 文件 (需要 `python-dotenv` 库并集成到应用中)。

**3.5. 运行应用程序**

使用 Uvicorn 启动 FastAPI 应用。确保 FastAPI 应用实例在 `app/main.py` 中名为 `app`。

```bash
# 在项目根目录，激活 .venv 环境后
# 示例启动命令
uv run uvicorn app.main:app --host 0.0.0.0 --port 8888 --workers 1
```

*   `--host 0.0.0.0`: 使服务可以从外部访问。
*   `--port 8888`: 服务监听的端口。
*   `--workers 1`: Uvicorn worker 的数量。对于 GPU 绑定的模型服务，通常设置为 1 worker/GPU，或者根据具体模型和并发需求调整。如果有多个 GPU 并希望每个 worker 使用不同 GPU，需要结合 `CUDA_VISIBLE_DEVICES` 等环境变量或在应用内进行设备分配。

**4. 服务持久化 (推荐)**

为了使服务在后台持续运行并在服务器重启后自动启动，建议使用 `systemd` 或 `supervisor`。

**4.1. 使用 `systemd` (示例)**

1.  创建一个 `systemd` 服务文件，例如 `/etc/systemd/system/sensevoice_asr.service`:

    ```ini
    [Unit]
    Description=SenseVoice ASR FastAPI Service
    After=network.target

    [Service]
    User=your_run_user # 替换为运行服务的用户，不建议用 root
    Group=your_run_group # 替换为运行服务的用户组
    WorkingDirectory=/srv/FunAudioLLM-SenseVoiceSmall # 项目根目录
    Environment="PATH=/srv/FunAudioLLM-SenseVoiceSmall/.venv/bin:/home/llm/miniconda3/bin:$PATH" # 确保虚拟环境的 python 和 uv 被找到
    # Environment="MODEL_PATH=/custom/model/path" # 设置其他环境变量
    # Environment="DEVICE=cuda:0"
    ExecStart=/srv/FunAudioLLM-SenseVoiceSmall/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
    Restart=always
    RestartSec=3

    [Install]
    WantedBy=multi-user.target
    ```

2.  **替换占位符**: `your_run_user`, `your_run_group`, 以及 `WorkingDirectory`, `ExecStart` 中的路径。
3.  **重载 systemd 配置**:
    ```bash
    sudo systemctl daemon-reload
    ```
4.  **启动服务**:
    ```bash
    sudo systemctl start sensevoice_asr
    ```
5.  **设置开机自启**:
    ```bash
    sudo systemctl enable sensevoice_asr
    ```
6.  **查看服务状态**:
    ```bash
    sudo systemctl status sensevoice_asr
    ```
7.  **查看服务日志**:
    ```bash
    sudo journalctl -u sensevoice_asr -f
    ```

**5. 测试部署**

服务启动后，可以使用 `curl` 从另一台机器或本地测试接口：

```bash
curl -X POST "http://127.0.0.1:8888/asr_pure" -F "file=@test_audio.wav"
```

**6. 更新服务**

1.  停止服务 (`sudo systemctl stop sensevoice_asr`)。
2.  进入项目目录 (`cd /srv/FunAudioLLM-SenseVoiceSmall`)。
3.  拉取最新代码 (例如 `git pull origin main`)。
4.  (如果需要) 激活虚拟环境 (`source .venv/bin/activate`) 并更新依赖 (`uv pip install -r requirements.txt` 或 `uv pip install .`)。
5.  启动服务 (`sudo systemctl start sensevoice_asr`)。

**7. 安全加固 (初步)**

*   **防火墙**: 确保只有必要的端口 (如 8000) 对外开放。
*   **非 root 用户**: 使用非 root 用户运行服务。
*   **HTTPS**: 对于生产环境，应考虑在 API 前面配置反向代理 (如 Nginx, Caddy) 并启用 HTTPS。

**8. 日志和监控**

*   **应用日志**: 应用日志的配置在 `detailed_design.md` 中提及。`systemd` 会捕获标准输出/错误流到 `journald`。
*   **系统监控**: 监控服务器的 CPU, GPU (使用 `nvidia-smi`), 内存, 磁盘使用情况。
