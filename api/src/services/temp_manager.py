"""Temporary file writer for audio downloads with Redis-based lifecycle management and S3 support"""

import asyncio
import os
import time
import uuid
from redis import asyncio as aioredis
from typing import Optional

import aiofiles
from loguru import logger

from ..core.config import settings

# Lua script for atomic cleanup: ZRANGEBYSCORE to find expired files, then ZREM to delete them
CLEANUP_LUA_SCRIPT = """
local zset_key = KEYS[1]
local current_time = tonumber(ARGV[1])
local batch_size = tonumber(ARGV[2])

-- Get expired members (score < current_time)
local expired = redis.call('ZRANGEBYSCORE', zset_key, '-inf', current_time, 'LIMIT', 0, batch_size)

-- Remove them atomically
if #expired > 0 then
    redis.call('ZREM', zset_key, unpack(expired))
end

return expired
"""


async def register_temp_file(redis: Optional["aioredis.Redis"], file_path: str, ttl_seconds: int, user_id: Optional[str] = None, is_s3: bool = False) -> None:
    """Register a temp file in Redis with expiry timestamp and user ownership
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file (local) or S3 key (S3)
        ttl_seconds: Time to live in seconds
        user_id: User ID who owns this file (for access control)
        is_s3: True if file_path is an S3 key, False if local file
    """
    if not redis:
        return
    
    try:
        expiry_timestamp = time.time() + ttl_seconds
        
        # Store file path with expiry in sorted set
        await redis.zadd(settings.temp_redis_zset_key, {file_path: expiry_timestamp})
        
        # Store user ownership in hash if user_id provided
        if user_id:
            ownership_key = f"{settings.temp_redis_zset_key}:ownership"
            await redis.hset(ownership_key, file_path, user_id)
            await redis.expire(ownership_key, ttl_seconds + 3600)  # Add buffer time
        
        # Store S3 flag in hash to track storage type
        storage_type_key = f"{settings.temp_redis_zset_key}:storage_type"
        storage_type = "s3" if is_s3 else "local"
        await redis.hset(storage_type_key, file_path, storage_type)
        await redis.expire(storage_type_key, ttl_seconds + 3600)  # Add buffer time
            
        logger.debug(f"Registered temp file in Redis: {file_path} (expires at {expiry_timestamp}, user: {user_id}, storage: {storage_type})")
    except Exception as e:
        logger.warning(f"Failed to register temp file in Redis: {e}")


async def remove_temp_registration(redis: Optional["aioredis.Redis"], file_path: str) -> None:
    """Remove temp file registration from Redis (call when download starts)
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file (local) or S3 key (S3)
    """
    if not redis:
        return
    
    try:
        await redis.zrem(settings.temp_redis_zset_key, file_path)
        
        # Also remove ownership record
        ownership_key = f"{settings.temp_redis_zset_key}:ownership"
        await redis.hdel(ownership_key, file_path)
        
        # Also remove storage type record
        storage_type_key = f"{settings.temp_redis_zset_key}:storage_type"
        await redis.hdel(storage_type_key, file_path)
        
        logger.debug(f"Removed temp file registration from Redis: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temp file from Redis: {e}")


async def check_file_exists_in_redis(redis: Optional["aioredis.Redis"], file_path: str) -> bool:
    """Check if temp file is still registered in Redis before I/O operation
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file (local) or S3 key (S3)
        
    Returns:
        True if file exists in Redis, False if not found or Redis unavailable
    """
    if not redis:
        # Fallback to filesystem check only for local files
        if not settings.enable_s3_storage:
            import aiofiles.os
            return await aiofiles.os.path.exists(file_path)
        return False
    
    try:
        score = await redis.zscore(settings.temp_redis_zset_key, file_path)
        if score is None:
            logger.debug(f"Temp file not found in Redis: {file_path}")
            return False
        
        # Check if file is expired
        if score < time.time():
            logger.debug(f"Temp file expired in Redis: {file_path}")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Failed to check temp file in Redis: {e}")
        # Fallback to filesystem check only for local files
        if not settings.enable_s3_storage:
            import aiofiles.os
            return await aiofiles.os.path.exists(file_path)
        return False


async def verify_file_ownership(redis: Optional["aioredis.Redis"], file_path: str, user_id: str) -> bool:
    """Verify that a user owns a specific temp file
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file
        user_id: User ID to verify ownership
        
    Returns:
        True if user owns the file or Redis not configured, False otherwise
    """
    if not redis:
        # If Redis not configured, allow access (backward compatibility)
        logger.debug(f"Redis not configured, allowing access to {file_path}")
        return True
    
    try:
        ownership_key = f"{settings.temp_redis_zset_key}:ownership"
        stored_user_id = await redis.hget(ownership_key, file_path)
        
        if stored_user_id is None:
            logger.warning(f"No ownership record found for {file_path}")
            # Allow access if no ownership record (backward compatibility)
            return True
        
        if stored_user_id == user_id:
            logger.debug(f"User {user_id} verified as owner of {file_path}")
            return True
        else:
            logger.warning(f"User {user_id} attempted to access file owned by {stored_user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to verify file ownership: {e}")
        # On error, deny all access to be safe
        return False

async def redis_cleanup_once(redis: "aioredis.Redis", batch_size: int = 100) -> int:
    """Execute one cleanup cycle: find expired files, delete them from Redis and filesystem/S3
    
    Args:
        redis: Redis client
        batch_size: Maximum number of files to cleanup in one batch
        
    Returns:
        Number of files cleaned up
    """
    try:
        current_time = time.time()
        
        # Execute Lua script atomically
        expired_files = await redis.eval(
            CLEANUP_LUA_SCRIPT,
            1,  # number of keys
            settings.temp_redis_zset_key,  # KEYS[1]
            current_time,  # ARGV[1]
            batch_size,  # ARGV[2]
        )
        
        if not expired_files:
            return 0
        
        # Get storage types for all expired files
        storage_type_key = f"{settings.temp_redis_zset_key}:storage_type"
        storage_types = {}
        for file_path_bytes in expired_files:
            file_path = file_path_bytes.decode("utf-8") if isinstance(file_path_bytes, bytes) else file_path_bytes
            storage_type = await redis.hget(storage_type_key, file_path)
            storage_types[file_path] = storage_type or "local"
        
        # Delete files from filesystem or S3
        deleted_count = 0
        
        # Get S3 client if needed
        s3_client = None
        if settings.enable_s3_storage:
            s3_client = settings.get_s3_client()
        
        for file_path_bytes in expired_files:
            file_path = file_path_bytes.decode("utf-8") if isinstance(file_path_bytes, bytes) else file_path_bytes
            storage_type = storage_types.get(file_path, "local")
            
            try:
                if storage_type == "s3":
                    # Delete from S3
                    if s3_client:
                        from .s3_helper import delete_from_s3
                        if await delete_from_s3(s3_client, file_path):
                            logger.info(f"Deleted expired S3 temp file: {file_path}")
                            deleted_count += 1
                        else:
                            logger.warning(f"Failed to delete S3 temp file: {file_path}")
                    else:
                        logger.warning(f"S3 client not available for deleting: {file_path}")
                else:
                    # Delete from local filesystem
                    import aiofiles.os
                    if await aiofiles.os.path.exists(file_path):
                        await aiofiles.os.remove(file_path)
                        logger.info(f"Deleted expired temp file: {file_path}")
                        deleted_count += 1
                    else:
                        logger.debug(f"Temp file already deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        # Clean up storage type records
        if expired_files:
            await redis.hdel(storage_type_key, *[
                fp.decode("utf-8") if isinstance(fp, bytes) else fp 
                for fp in expired_files
            ])
        
        return deleted_count
    except Exception as e:
        logger.error(f"Redis cleanup cycle failed: {e}")
        return 0


async def redis_periodic_cleanup_loop(redis: "aioredis.Redis") -> None:
    """Background task that periodically cleans up expired temp files using Redis
    
    Args:
        redis: Redis client
    """
    interval = settings.temp_redis_cleanup_interval_seconds
    batch_size = settings.temp_cleaner_batch_size
    
    logger.info(f"Starting Redis temp file cleanup loop (interval={interval}s, batch_size={batch_size})")
    
    while True:
        try:
            deleted_count = await redis_cleanup_once(redis, batch_size)
            if deleted_count > 0:
                logger.info(f"Redis cleanup cycle deleted {deleted_count} files")
        except asyncio.CancelledError:
            logger.info("Redis cleanup loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Redis cleanup loop: {e}")
        
        await asyncio.sleep(interval)


async def cleanup_temp_files() -> None:
    """Filesystem-based cleanup for temp files
    
    This is a deterministic cleanup that:
    1. Deletes files older than max_temp_dir_age_hours
    2. Deletes oldest files if count exceeds max_temp_dir_count
    3. Deletes oldest files if total size exceeds max_temp_dir_size_mb
    
    Note: Only runs if filesystem temp cleanup is enabled and S3 storage is disabled
    """
    if not settings.enable_temp_file_system or settings.enable_s3_storage:
        logger.debug("Filesystem temp cleanup disabled or S3 storage enabled")
        return
    
    try:
        import aiofiles.os
        
        if not await aiofiles.os.path.exists(settings.temp_file_dir):
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            return

        # Get all temp files with stats
        files = []
        total_size = 0

        # Use os.scandir for sync iteration, but aiofiles.os.stat for async stats
        for entry in os.scandir(settings.temp_file_dir):
            if entry.is_file():
                stat = await aiofiles.os.stat(entry.path)
                files.append((entry.path, stat.st_mtime, stat.st_size))
                total_size += stat.st_size

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])

        # Get current time
        current_time = time.time()
        max_age = settings.max_temp_dir_age_hours * 3600

        deleted_count = 0
        for path, mtime, size in files:
            should_delete = False

            # Check age
            if current_time - mtime > max_age:
                should_delete = True
                logger.info(f"Deleting old temp file: {path}")

            # Check count limit
            elif len(files) - deleted_count > settings.max_temp_dir_count:
                should_delete = True
                logger.info(f"Deleting excess temp file: {path}")

            # Check size limit
            elif total_size > settings.max_temp_dir_size_mb * 1024 * 1024:
                should_delete = True
                logger.info(f"Deleting to reduce directory size: {path}")

            if should_delete:
                try:
                    await aiofiles.os.remove(path)
                    total_size -= size
                    deleted_count += 1
                    logger.info(f"Deleted temp file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")

    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")


class TempFileWriter:
    """Handles writing audio chunks to a temp file with Redis lifecycle management and S3 support"""

    def __init__(self, format: str, redis: Optional["aioredis.Redis"] = None, user_id: Optional[str] = None, s3_client=None, return_s3_key: bool = False):
        """Initialize temp file writer

        Args:
            format: Audio format extension (mp3, wav, etc)
            redis: Optional Redis client for lifecycle management
            user_id: Optional user ID for ownership tracking
            s3_client: Optional S3 client (boto3) for S3 storage
            return_s3_key: If True, return S3 key with signature instead of download link
        """
        self.format = format
        self.redis = redis
        self.user_id = user_id
        self.s3_client = s3_client
        self.return_s3_key = return_s3_key
        self.temp_file = None
        self._finalized = False
        self._write_error = False  # Flag to track if we've had a write error
        self._use_s3 = settings.enable_s3_storage and s3_client is not None
        self._buffer = bytearray()  # Buffer for S3 uploads
        self.s3_key_data = None  # Store S3 key data for return_s3_key

    async def __aenter__(self):
        """Async context manager entry"""
        try:
            # Generate UUID-based identifier
            file_uuid = uuid.uuid4().hex
            
            if self._use_s3:
                # S3 mode: store in buffer, upload on finalize
                self.s3_key = f"temp/{file_uuid}.{self.format}"
                self.temp_path = self.s3_key  # Use S3 key as path
                
                # Generate S3 key data with signature
                from .s3_helper import create_s3_key_with_signature
                self.s3_key_data = create_s3_key_with_signature(self.s3_key)
                
                # Generate download path with S3 key info
                self.download_path = f"/download/s3/{self.s3_key_data['key']}?signature={self.s3_key_data['signature']}"
                
                logger.debug(f"Created S3 temp key: {self.s3_key} for user: {self.user_id}")
            else:
                # Local filesystem mode
                import aiofiles.os
                
                # Clean up old files first (only if filesystem cleanup is enabled)
                if settings.enable_temp_file_system:
                    await cleanup_temp_files()

                # Create temp file with UUID-based name
                await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
                
                filename = f"{file_uuid}.{self.format}"
                self.temp_path = os.path.join(settings.temp_file_dir, filename)
                
                # Create and open file
                self.temp_file = await aiofiles.open(self.temp_path, mode="wb")

                # Generate download path immediately
                self.download_path = f"/download/{os.path.basename(self.temp_path)}"
                
                logger.debug(f"Created temp file: {self.temp_path} for user: {self.user_id}")
            
            # Register in Redis if available (with user_id for ownership)
            if self.redis:
                await register_temp_file(
                    self.redis, 
                    self.temp_path, 
                    settings.temp_file_ttl_seconds, 
                    self.user_id,
                    is_s3=self._use_s3
                )
                
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to create temp file/S3 key: {e}")
            self._write_error = True
            # Set a placeholder path so the API can still function
            self.temp_path = f"unavailable_{self.format}"
            self.download_path = f"/download/{self.temp_path}"

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.temp_file and not self._finalized:
                await self.temp_file.close()
                self._finalized = True
        except Exception as e:
            logger.error(f"Error closing temp file: {e}")
            self._write_error = True

    async def write(self, chunk: bytes) -> None:
        """Write a chunk of audio data

        Args:
            chunk: Audio data bytes to write
        """
        if self._finalized:
            raise RuntimeError("Cannot write to finalized temp file")

        # Skip writing if we've already encountered an error
        if self._write_error:
            return

        try:
            if self._use_s3:
                # Buffer data for S3 upload
                self._buffer.extend(chunk)
            else:
                # Write to local file
                if not self.temp_file:
                    return
                await self.temp_file.write(chunk)
                await self.temp_file.flush()
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to write to temp file: {e}")
            self._write_error = True

    async def finalize(self) -> str:
        """Close temp file and return download path

        Returns:
            Path to use for downloading the temp file
        """
        if self._finalized:
            raise RuntimeError("Temp file already finalized")

        # Skip finalizing if we've already encountered an error
        if self._write_error:
            self._finalized = True
            return self.download_path

        try:
            if self._use_s3:
                # Upload buffer to S3
                if self._buffer and self.s3_client:
                    from .s3_helper import upload_to_s3
                    success = await upload_to_s3(self.s3_client, self.s3_key, bytes(self._buffer))
                    if not success:
                        logger.error(f"Failed to upload to S3: {self.s3_key}")
                        self._write_error = True
                    else:
                        logger.info(f"Uploaded {len(self._buffer)} bytes to S3: {self.s3_key}")
                    # Clear buffer
                    self._buffer.clear()
            else:
                # Close local file
                if self.temp_file:
                    await self.temp_file.close()
                    
            self._finalized = True
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to finalize temp file: {e}")
            self._write_error = True
            self._finalized = True

        return self.download_path
