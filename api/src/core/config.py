import torch
import redis.asyncio as aioredis
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880
    api_url_prefix: str = ""  # Optional URL prefix for all routes (e.g., "/api", "/v2")
    server_base_url: str | None = None  # Base URL for generating full download links (e.g., "http://localhost:8880")
    
    # Security Settings
    hide_server_header: bool = True  # Hide server header from responses
    api_bearer_token: str | None = None  # Optional Bearer token for authentication
    
    # API Documentation Settings
    enable_openapi_docs: bool = True  # Whether to enable OpenAPI documentation (/docs, /redoc, /openapi.json)

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep
    enable_temp_file_system: bool = True  # Enable temp file system (set to false if using Redis)
    
    # Redis Settings for Temp File Management
    redis_host: str | None = None  # Redis host (e.g., localhost)
    redis_port: int | None = None  # Redis port (e.g., 6379)
    redis_username: str | None = None  # Optional Redis username for ACL
    redis_password: str | None = None  # Optional Redis password
    temp_redis_zset_key: str = "temp_files"  # Redis sorted set key for temp files
    temp_file_ttl_seconds: int = 3600  # TTL for temp files in seconds (default: 1 hour)
    temp_redis_cleanup_interval_seconds: int = 60  # Cleanup interval in seconds
    temp_cleaner_batch_size: int = 100  # Number of files to clean per batch

    class Config:
        env_file = ".env"

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_base_url(self) -> str:
        """Get the base URL for generating full download links
        
        Returns the configured server_base_url or constructs one from host:port
        """
        if self.server_base_url:
            return self.server_base_url.rstrip("/")
        
        # Construct from host and port
        # Use localhost if host is 0.0.0.0
        host_for_url = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host_for_url}:{self.port}"

    def get_redis_client(self) -> "aioredis.Redis | None":
        """Get configured Redis client for temp file management
        
        Returns None if Redis is not configured or not available
        """
        # Require host and port for Redis configuration
        if not self.redis_host or not self.redis_port:
            return None

        try:
            from loguru import logger

            # Build Redis connection kwargs
            kwargs = {"decode_responses": True}

            # Add username/password if provided
            if self.redis_username:
                kwargs["username"] = self.redis_username
            if self.redis_password:
                kwargs["password"] = self.redis_password

            # Construct URL from host/port (no DB index by default)
            redis_url = f"redis://{self.redis_host}:{self.redis_port}"
            return aioredis.from_url(redis_url, **kwargs)
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to create Redis client: {e}")
            return None


settings = Settings()
