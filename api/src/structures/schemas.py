from email.policy import default
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class VoiceCombineRequest(BaseModel):
    """Request schema for voice combination endpoint that accepts either a string with + or a list"""

    voices: Union[str, List[str]] = Field(
        ...,
        description="Either a string with voices separated by + (e.g. 'voice1+voice2') or a list of voice names to combine",
    )


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# OpenAI-compatible schemas
class WordTimestamp(BaseModel):
    """Word-level timestamp information"""

    word: str = Field(..., description="The word or token")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class CaptionedSpeechResponse(BaseModel):
    """Response schema for captioned speech endpoint"""

    audio: str = Field(..., description="The generated audio data encoded in base 64")
    audio_format: str = Field(..., description="The format of the output audio")
    timestamps: Optional[List[WordTimestamp]] = Field(
        ..., description="Word-level timestamps"
    )


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words"
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default = 1.0,
        description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )


# Response Models for OpenAPI type safety


class ModelObject(BaseModel):
    """Model object schema"""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = Field(..., description="Organization that owns the model")


class ModelsListResponse(BaseModel):
    """Response schema for models list endpoint"""

    object: str = Field(default="list", description="Object type")
    data: List[ModelObject] = Field(..., description="List of available models")


class VoicesListResponse(BaseModel):
    """Response schema for voices list endpoint"""

    voices: List[str] = Field(..., description="List of available voice names")


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint"""

    status: str = Field(..., description="Health status of the API")


class TestEndpointResponse(BaseModel):
    """Response schema for test endpoint"""

    status: str = Field(..., description="Test endpoint status")


class ErrorResponse(BaseModel):
    """Generic error response schema"""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ThreadInfo(BaseModel):
    """Thread information schema"""

    name: str = Field(..., description="Thread name")
    id: int = Field(..., description="Thread identifier")
    alive: bool = Field(..., description="Whether thread is alive")
    daemon: bool = Field(..., description="Whether thread is daemon")


class ThreadsResponse(BaseModel):
    """Response schema for threads debug endpoint"""

    total_threads: int = Field(..., description="Total number of threads")
    active_threads: int = Field(..., description="Number of active threads")
    thread_names: List[str] = Field(..., description="List of thread names")
    thread_details: List[ThreadInfo] = Field(..., description="Detailed thread information")
    memory_mb: float = Field(..., description="Memory usage in MB")


class StorageInfo(BaseModel):
    """Storage directory information schema"""

    path: str = Field(..., description="Directory path")
    exists: bool = Field(..., description="Whether directory exists")
    total_files: int = Field(..., description="Total number of files")
    total_size_mb: float = Field(..., description="Total size in MB")
    oldest_file_age_hours: float | None = Field(..., description="Age of oldest file in hours")


class StorageResponse(BaseModel):
    """Response schema for storage debug endpoint"""

    temp_dir: StorageInfo = Field(..., description="Temporary directory information")
    output_dir: StorageInfo = Field(..., description="Output directory information")


class DiskPartitionInfo(BaseModel):
    """Disk partition information schema"""

    device: str = Field(..., description="Device identifier")
    mountpoint: str = Field(..., description="Mount point path")
    fstype: str = Field(..., description="File system type")
    total_gb: float = Field(..., description="Total capacity in GB")
    used_gb: float = Field(..., description="Used space in GB")
    free_gb: float = Field(..., description="Free space in GB")
    percent_used: float = Field(..., description="Percentage of space used")


class DiskStorageResponse(BaseModel):
    """Response schema for disk storage endpoint"""

    storage_info: List[DiskPartitionInfo] = Field(..., description="List of disk partition information")


class GPUInfo(BaseModel):
    """GPU information schema"""

    id: int = Field(..., description="GPU identifier")
    name: str = Field(..., description="GPU name")
    memory_total: float = Field(..., description="Total memory in MB")
    memory_used: float = Field(..., description="Used memory in MB")
    memory_free: float = Field(..., description="Free memory in MB")
    memory_percent: float = Field(..., description="Memory usage percentage")
    gpu_util: float = Field(..., description="GPU utilization percentage")
    temperature: float = Field(..., description="GPU temperature in Celsius")


class DetailedCPUInfo(BaseModel):
    """Detailed CPU information schema"""

    cpu_count: int | None = Field(None, description="Number of CPU cores")
    cpu_percent: float = Field(..., description="Overall CPU usage percentage")
    per_cpu_percent: List[float] = Field(..., description="Per-CPU usage percentages")
    load_avg: Tuple[float, float, float] | List[float] = Field(..., description="Load average (1, 5, 15 min)")


class VirtualMemoryInfo(BaseModel):
    """Virtual memory information schema"""

    total_gb: float = Field(..., description="Total memory in GB")
    available_gb: float = Field(..., description="Available memory in GB")
    used_gb: float = Field(..., description="Used memory in GB")
    percent: float = Field(..., description="Memory usage percentage")


class SwapMemoryInfo(BaseModel):
    """Swap memory information schema"""

    total_gb: float = Field(..., description="Total swap in GB")
    used_gb: float = Field(..., description="Used swap in GB")
    free_gb: float = Field(..., description="Free swap in GB")
    percent: float = Field(..., description="Swap usage percentage")


class DetailedMemoryInfo(BaseModel):
    """Detailed memory information schema"""

    virtual: VirtualMemoryInfo = Field(..., description="Virtual memory information")
    swap: SwapMemoryInfo = Field(..., description="Swap memory information")


class ProcessInfo(BaseModel):
    """Process information schema"""

    pid: int = Field(..., description="Process ID")
    status: str = Field(..., description="Process status")
    create_time: str = Field(..., description="Process creation time (ISO format)")
    cpu_percent: float = Field(..., description="Process CPU usage percentage")
    memory_percent: float = Field(..., description="Process memory usage percentage")


class NetworkIOCounters(BaseModel):
    """Network IO counters schema"""
    
    class Config:
        extra = "allow"  # Allow additional fields from psutil


class NetworkInfo(BaseModel):
    """Network information schema"""

    connections: int = Field(..., description="Number of network connections")
    network_io: dict = Field(..., description="Network IO counters")


class DetailedGPUMemory(BaseModel):
    """Detailed GPU memory information schema"""

    total: float = Field(..., description="Total GPU memory")
    used: float = Field(..., description="Used GPU memory")
    free: float = Field(..., description="Free GPU memory")
    percent: float = Field(..., description="GPU memory usage percentage")


class DetailedGPUInfo(BaseModel):
    """Detailed GPU information schema for system endpoint"""

    id: int = Field(..., description="GPU identifier")
    name: str = Field(..., description="GPU name")
    load: float = Field(..., description="GPU load")
    memory: DetailedGPUMemory = Field(..., description="GPU memory information")
    temperature: float = Field(..., description="GPU temperature in Celsius")


class MPSGPUInfo(BaseModel):
    """MPS (Apple Silicon) GPU information schema"""

    type: str = Field(..., description="GPU type")
    available: bool = Field(..., description="Whether GPU is available")
    device: str = Field(..., description="Device name")
    backend: str = Field(..., description="Backend technology")


class DetailedSystemResponse(BaseModel):
    """Detailed response schema for system debug endpoint"""

    cpu: DetailedCPUInfo = Field(..., description="CPU information")
    memory: DetailedMemoryInfo = Field(..., description="Memory information")
    process: ProcessInfo = Field(..., description="Process information")
    network: NetworkInfo = Field(..., description="Network information")
    gpu: List[DetailedGPUInfo] | MPSGPUInfo | str | None = Field(None, description="GPU information")


class SystemResponse(BaseModel):
    """Response schema for system debug endpoint"""

    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    memory_available_mb: float = Field(..., description="Available memory in MB")
    memory_used_mb: float = Field(..., description="Used memory in MB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    disk_free_gb: float = Field(..., description="Free disk space in GB")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpus: List[GPUInfo] | None = Field(default=None, description="List of GPU information")


class SessionInfo(BaseModel):
    """Session information schema"""

    model: str = Field(..., description="Model path")
    age_seconds: float = Field(..., description="Age of session in seconds")


class GPUSessionInfo(BaseModel):
    """GPU session information schema"""

    model: str = Field(..., description="Model path")
    age_seconds: float = Field(..., description="Age of session in seconds")
    stream_id: int = Field(..., description="CUDA stream ID")


class GPUMemoryInfo(BaseModel):
    """GPU memory information for session pools"""

    total_mb: float = Field(..., description="Total GPU memory in MB")
    used_mb: float = Field(..., description="Used GPU memory in MB")
    free_mb: float = Field(..., description="Free GPU memory in MB")
    percent_used: float = Field(..., description="Percentage of GPU memory used")


class CPUPoolInfo(BaseModel):
    """CPU session pool information"""

    active_sessions: int = Field(..., description="Number of active sessions")
    max_sessions: int = Field(..., description="Maximum number of sessions")
    sessions: List[SessionInfo] = Field(..., description="List of active sessions")


class GPUPoolInfo(BaseModel):
    """GPU session pool information"""

    active_sessions: int = Field(..., description="Number of active sessions")
    max_streams: int = Field(..., description="Maximum number of streams")
    available_streams: int = Field(..., description="Number of available streams")
    sessions: List[GPUSessionInfo] = Field(..., description="List of active sessions")
    memory: Optional[GPUMemoryInfo] = Field(None, description="GPU memory information")


class SessionPoolsResponse(BaseModel):
    """Response schema for session pools debug endpoint"""

    cpu: Optional[CPUPoolInfo] = Field(None, description="CPU pool information")
    gpu: Optional[GPUPoolInfo] = Field(None, description="GPU pool information")


class CaptionedSpeechRequest(BaseModel):
    """Request schema for captioned speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_timestamps: bool = Field(
        default=True,
        description="If true (default), returns word-level timestamps in the response",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default = 1.0,
        description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
