**架构设计文档**

**1. 系统概述**

本项目旨在实现一个基于 FunAudioLLM/SenseVoiceSmall 模型的语音转录文本服务。该服务将通过 FastAPI 构建的 RESTful API 对外提供，用户可以通过上传音频文件获取转录文本。系统将部署在配备 NVIDIA GPU 的 Ubuntu 服务器上，并使用 `uv` 进行 Python 环境管理。

**2. 组件设计**

系统主要包含以下几个核心组件：

*   **API 接口层 (FastAPI)**：负责接收用户请求，校验输入，并将任务分发给语音识别服务层。
*   **语音识别服务层 (ASR Service)**：核心业务逻辑层，负责加载 SenseVoiceSmall 模型，对输入的音频进行预处理、执行语音识别，并返回结果。
*   **模型管理模块**：负责 SenseVoiceSmall 模型的加载、卸载以及推理资源（如 GPU）的分配和管理。
*   **环境与部署**：定义项目的运行环境、依赖管理和部署方式。

**3. 详细设计**

**3.1. API 接口层**

*   **框架**: FastAPI
*   **Web 服务器**: Uvicorn
*   **端点 (Endpoint)**:
    *   `POST /asr_pure`
        *   **请求**: `multipart/form-data`，包含一个名为 `file` 的音频文件字段 (例如：`.wav`, `.mp3`)。
        *   **成功响应 (200 OK)**: JSON 对象，包含转录文本。
            ```json
            {
              "text": "转录后的文本内容...",
              "status": "success",
              "processing_time_ms": 1234
            }
            ```
        *   **错误响应**:
            *   `400 Bad Request`: 输入文件无效或格式不支持。
                ```json
                {
                  "detail": "无效的音频文件或格式不支持。",
                  "status": "error"
                }
                ```
            *   `422 Unprocessable Entity`: FastAPI 自动处理请求体校验失败。
            *   `500 Internal Server Error`: 服务器内部错误，例如模型加载失败或识别过程中发生未知异常。
                ```json
                {
                  "detail": "语音识别服务内部错误。",
                  "status": "error"
                }
                ```
*   **依赖**: `fastapi`, `uvicorn`, `python-multipart` (处理文件上传)

**3.2. 语音识别服务层 (ASR Service)**

*   **核心功能**:
    1.  接收 API 层传递过来的音频文件。
    2.  **音频预处理**:
        *   验证音频文件格式和完整性。
        *   (可选) 将音频转换为模型期望的格式/采样率（如果模型有特定要求）。
    3.  **模型推理**:
        *   调用模型管理模块获取 SenseVoiceSmall 模型实例。
        *   将预处理后的音频数据输入模型进行语音识别。
    4.  **结果后处理**:
        *   获取模型输出的原始文本。
        *   (可选) 进行文本规范化，如去除不必要的空格、标点符号校正等。
*   **模型**: FunAudioLLM/SenseVoiceSmall
    *   **本地路径**: `/home/llm/model/iic/SenseVoiceSmall`
*   **依赖**:
    *   `torch`, `torchaudio` (或模型依赖的特定音频处理库)
    *   `transformers` (如果模型通过 Hugging Face Transformers 加载)
    *   模型本身的其他特定依赖。

**3.3. 模型管理模块**

*   **职责**:
    *   在服务启动时加载 SenseVoiceSmall 模型到指定的 GPU 设备。考虑到有 4 块 2080Ti，可以设计为将模型副本加载到一块或多块 GPU 上，或者根据请求量动态分配。
    *   提供获取模型实例的接口给 ASR 服务层。
    *   管理 GPU 资源，确保有效利用。
*   **策略**:
    *   **模型加载**: 服务启动时预加载模型到内存和显存，以减少单次请求的延迟。
    *   **多 GPU**:
        *   **方案一 (简单)**: 选择一块 GPU 运行模型。可以通过环境变量 (如 `CUDA_VISIBLE_DEVICES`) 或在代码中指定。
        *   **方案二 (并行)**: 如果模型和代码支持，可以考虑在多个 GPU 上并行处理请求，或者将模型分片到多个 GPU（如果模型巨大且支持）。对于 SenseVoiceSmall 这种小型模型，单个 GPU 处理请求可能已经足够高效，多 GPU 可以用于处理并发请求，每个 GPU 处理一个独立的请求。
*   **配置**: 允许配置使用的 GPU ID(s)。

**3.4. 环境与部署**

*   **操作系统**: Ubuntu 24.04
*   **Python 版本**: 使用 `/home/llm/miniconda3/bin/python` (建议通过 `uv` 创建独立的虚拟环境)。
*   **Python 环境管理**: `uv`
    *   使用 `pyproject.toml` (推荐) 或 `requirements.txt` 管理项目依赖。
*   **CUDA 版本**: 12.4
*   **NVIDIA 驱动**: 550.144.03
*   **部署**:
    1.  使用 `uv` 创建和激活虚拟环境。
    2.  使用 `uv pip install -r requirements.txt` (或 `uv pip install .` 如果使用 `pyproject.toml`) 安装依赖。
    3.  使用 Uvicorn 启动 FastAPI 应用：`uvicorn main:app --host 0.0.0.0 --port 8000` (其中 `main.py` 是 FastAPI 应用入口文件)。
    4.  可以考虑使用 `systemd` 或 `supervisor` 将服务作为后台守护进程运行。

**4. 硬件利用**

*   **CPU**: 56 核 CPU，主要用于 FastAPI 请求处理、数据预处理/后处理等非模型核心运算任务。
*   **GPU**: 4x Nvidia 2080Ti (共 88G 显存)。SenseVoiceSmall 模型相对较小，单张 2080Ti (22G) 显存足以容纳。
    *   可以配置服务运行在单个 GPU 上，或者为每个 FastAPI worker 分配一个 GPU 实例（如果并发量大）。
*   **内存**: 512G 物理内存，足以缓存模型和处理大量并发请求的数据。
*   **磁盘**: 2T SSD，用于存储操作系统、模型文件、日志等。

**5. 项目结构 (建议)**

```
FunAudioLLM-SenseVoiceSmall/
├── app/                      # FastAPI 应用代码
│   ├── __init__.py
│   ├── main.py               # FastAPI 应用入口和路由定义
│   ├── services/             # 业务逻辑，如 ASR 服务
│   │   ├── __init__.py
│   │   └── asr_service.py
│   ├── models/               # 模型加载和管理
│   │   ├── __init__.py
│   │   └── sensevoice_loader.py
│   └── schemas.py            # Pydantic 模型，用于请求和响应数据校验
├── pyproject.toml            # 或 requirements.txt，项目依赖和 uv 配置
├── tests/                    # 测试代码
├── scripts/                  # 辅助脚本 (如启动脚本)
└── README.md
```

**6. 非功能性需求**

*   **性能**: 尽可能低的语音转录延迟。通过预加载模型、优化音频处理流程来实现。
*   **可扩展性**: 当前设计主要针对单机多 GPU。未来若需更高并发，可考虑使用如 Kubernetes 进行水平扩展，将服务部署到多个节点。
*   **可靠性**: 充分的错误处理和日志记录。
*   **可维护性**: 清晰的代码结构和文档。

**7. 后续步骤**

1.  **环境搭建**: 使用 `uv` 创建虚拟环境。
2.  **安装核心依赖**: `fastapi`, `uvicorn`, `torch`, `torchaudio`, `transformers` (以及 SenseVoice 所需的其他库)。
3.  **实现模型加载模块**: 能够从 `/home/llm/model/iic/SenseVoiceSmall` 加载模型并进行简单推理测试。
4.  **实现 ASR 服务层**: 封装模型推理逻辑。
5.  **实现 API 接口层**: 使用 FastAPI 定义 `/asr_pure` 接口。
6.  **编写测试用例**。
7.  **部署和测试**。 