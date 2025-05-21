**详细设计文档**

**1. 引言**

本文档基于《架构设计文档》(architecture.md) 进行详细阐述，旨在为 FunAudioLLM/SenseVoiceSmall 语音转录服务的开发提供具体指导。文档将深入描述各个模块的内部设计、数据流、接口规范以及关键的错误处理策略。

**2. 系统组件详细设计**

**2.1. API 接口层 (FastAPI - `app/main.py`)**

*   **主要职责**:
    *   接收 HTTP POST 请求到 `/asr_pure`。
    *   使用 Pydantic 模型 (`app/schemas.py`) 对请求进行校验。
    *   调用 `ASRService` 处理语音文件。
    *   格式化 `ASRService` 的响应并返回给客户端。
    *   统一的异常处理中间件，将业务异常和未知错误转换为标准的 JSON 错误响应。
*   **端点 `/asr_pure`**:
    *   **请求处理流程**:
        1.  接收到 `multipart/form-data` 请求。
        2.  FastAPI 自动解析 `file: UploadFile` 参数。
        3.  检查文件类型 (例如, 允许 `audio/wav`, `audio/mpeg`) 和大小限制 (可配置)。
        4.  将 `UploadFile` 对象传递给 `ASRService.transcribe_audio_file()`。
        5.  获取转录结果或异常。
        6.  若成功，返回 `schemas.ASRResponse`。
        7.  若失败，根据异常类型返回相应的错误 `schemas.ErrorResponse`。
*   **数据模型 (`app/schemas.py`)**:
    *   `ASRResponse(BaseModel)`:
        *   `text: str`
        *   `status: str = "success"`
        *   `processing_time_ms: Optional[float] = None`
    *   `ErrorResponse(BaseModel)`:
        *   `detail: str`
        *   `status: str = "error"`
*   **错误处理**:
    *   FastAPI 的 `RequestValidationError` (422) 会自动处理。
    *   自定义异常处理器 (Exception Handler):
        *   `AudioProcessingError` (自定义业务异常): 返回 400 或 500，取决于错误原因。
        *   `ModelInferenceError` (自定义业务异常): 返回 500。
        *   通用 `Exception`: 返回 500，记录详细日志。
*   **配置**:
    *   允许的最大文件大小。
    *   支持的音频 MIME 类型。

**2.2. 语音识别服务层 (`app/services/asr_service.py`) - ASRService**

*   **主要职责**:
    *   提供核心的语音转录逻辑。
    *   与 `SenseVoiceLoader` 交互获取模型实例。
    *   执行音频预处理、模型推理和结果后处理。
*   **核心方法**:
    *   `async def transcribe_audio_file(self, audio_file: UploadFile) -> str:`
        1.  **保存/读取音频**:
            *   从 `UploadFile` 对象读取音频数据。可以先保存到临时文件，或直接在内存中处理（注意内存消耗）。
            *   记录处理开始时间。
        2.  **音频预处理**:
            *   使用 `torchaudio` 或类似库加载音频。
            *   检查采样率、通道数。如果模型有特定要求（如16kHz单通道），则进行转换。
            *   将音频数据转换为模型所需的张量格式。
            *   **异常处理**: 若音频无效或无法处理，抛出 `AudioProcessingError`。
        3.  **模型推理**:
            *   调用 `self.model_loader.get_model()` 获取模型。
            *   调用 `self.model_loader.get_device()` 获取设备 (e.g., 'cuda:0')。
            *   将预处理后的音频张量移至指定设备。
            *   执行模型推理: `model(audio_tensor)`。
            *   **异常处理**: 若模型推理失败，抛出 `ModelInferenceError`。
        4.  **结果后处理**:
            *   从模型输出中提取转录文本。
            *   (可选) 进行文本清洗、规范化。
            *   记录处理结束时间，计算 `processing_time_ms`。
        5.  返回转录文本。
*   **依赖注入**:
    *   构造函数接收 `SenseVoiceLoader` 实例。
*   **自定义异常**:
    *   `AudioProcessingError(Exception)`: 音频文件处理相关错误。
    *   `ModelInferenceError(Exception)`: 模型推理阶段错误。

**2.3. 模型管理模块 (`app/models/sensevoice_loader.py`) - SenseVoiceLoader**

*   **主要职责**:
    *   加载和管理 SenseVoiceSmall 模型。
    *   提供模型实例和运行设备的访问。
*   **配置**:
    *   `MODEL_PATH: str = "/home/llm/model/iic/SenseVoiceSmall"` (可通过环境变量覆盖)
    *   `DEVICE: str = "cuda:0"` (可通过配置选择，或实现更复杂的 GPU 分配逻辑)
*   **核心方法**:
    *   `__init__(self)`:
        *   记录日志，开始加载模型。
        *   根据 `MODEL_PATH` 加载 SenseVoiceSmall 模型 (具体加载方式取决于模型库，如 Hugging Face `from_pretrained` 或直接加载 PyTorch 模型文件)。
        *   将模型移至 `DEVICE`。
        *   `self.model.eval()` 设置为评估模式。
        *   记录日志，模型加载完成或失败。
        *   **异常处理**: 若模型加载失败 (路径错误、文件损坏、显存不足)，抛出 `ModelLoadError` 并终止服务启动或标记为不可用。
    *   `get_model(self) -> Any`: 返回加载的模型实例。
    *   `get_device(self) -> str`: 返回模型所在的设备字符串。
*   **单例模式 (推荐)**:
    *   确保整个应用只加载一次模型，避免资源浪费和重复加载。可以通过 FastAPI 的 `lifespan` 事件或依赖注入系统实现。
*   **多 GPU 策略 (如 `architecture.md` 中所述)**:
    *   **简单方案**: `DEVICE` 可配置为 `cuda:0`, `cuda:1` 等。
    *   **高级方案 (未来考虑)**: 如果使用多个 FastAPI worker，每个 worker 可以绑定到不同的 GPU，或者实现一个请求分发器将任务分配到不同 GPU 上的模型实例。SenseVoiceSmall 可能不需要如此复杂的设计。
*   **自定义异常**:
    *   `ModelLoadError(Exception)`: 模型加载失败。

**3. 数据流**

1.  **客户端 -> API 层**: 用户通过 `curl` 或其他 HTTP 客户端发送 `POST` 请求，携带音频文件到 `http://<host>:<port>/asr_pure`。
2.  **API 层 (FastAPI)**:
    *   接收请求，`UploadFile` 对象创建。
    *   调用 `ASRService.transcribe_audio_file(audio_file)`.
3.  **ASR 服务层 (`ASRService`)**:
    *   从 `SenseVoiceLoader` 获取模型和设备。
    *   读取 `audio_file` 内容。
    *   预处理音频数据 (转换格式、采样率等)。
    *   将数据送入 SenseVoice 模型进行推理。
    *   获取原始转录文本。
    *   后处理文本。
4.  **ASR 服务层 -> API 层**: 返回包含转录文本的字符串和处理时间。
5.  **API 层 -> 客户端**: 将结果封装为 JSON (成功或错误) 并返回给客户端。

**4. 关键技术点和决策**

*   **Python 环境**: `uv` 用于创建和管理独立的 Python 环境，确保依赖隔离。
*   **异步处理**: FastAPI 和 Uvicorn 本身支持异步操作。`transcribe_audio_file` 方法设计为 `async`，对于 IO 密集型操作（文件读写）和潜在的模型推理阻塞，可以更好地利用异步特性，提高并发处理能力。模型推理本身如果是 CPU 密集型且阻塞的，需要考虑在线程池中运行（FastAPI 默认会为同步函数做此处理）。
*   **配置管理**: 关键配置（模型路径、设备、日志级别等）应通过环境变量或配置文件进行管理，而不是硬编码。
*   **日志记录**: 使用 Python 内置的 `logging` 模块。
    *   在关键步骤（API 请求接收、模型加载、推理开始/结束、错误发生）记录详细日志。
    *   配置日志级别 (DEBUG, INFO, WARNING, ERROR)。
    *   日志可以输出到控制台和/或文件。

**5. 安全考虑 (初步)**

*   **输入校验**: 严格校验上传的文件类型和大小，防止恶意文件上传。
*   **资源限制**: 防止单个请求消耗过多服务器资源（例如，超大音频文件导致内存溢出）。
*   **依赖安全**: 定期更新依赖库，扫描已知漏洞。

**6. 扩展性与可维护性**

*   **模块化设计**: 各组件职责清晰，低耦合。
*   **配置驱动**: 方便调整参数和行为。
*   **清晰的 API**: 内部和外部接口定义明确。

**7. 未来可能的增强**

*   **更复杂的 GPU 管理**: 动态分配 GPU 资源，支持模型副本在多 GPU 运行。
*   **批处理 API**: 支持一次上传多个音频文件或一个包含多个音频的压缩包。
*   **WebSockets**: 对于长音频，可以考虑使用 WebSockets 实现流式语音识别。
*   **身份验证/授权**: 如果需要限制 API 访问。
*   **更详细的指标监控**: 集成 Prometheus 等监控系统。 