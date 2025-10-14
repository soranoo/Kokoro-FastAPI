from email.policy import default
from enum import Enum
from typing import List, Literal, Optional, Union

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


class SessionPoolsResponse(BaseModel):
    """Response schema for session pools debug endpoint"""

    message: str = Field(..., description="Status message")


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
