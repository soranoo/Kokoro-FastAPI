"""Temporary file writer for audio downloads with Redis-based lifecycle management"""

import asyncio
import os
import time
import uuid
import redis.asyncio as aioredis
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


async def register_temp_file(redis: Optional["aioredis.Redis"], file_path: str, ttl_seconds: int) -> None:
    """Register a temp file in Redis with expiry timestamp
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file
        ttl_seconds: Time to live in seconds
    """
    if not redis:
        return
    
    try:
        expiry_timestamp = time.time() + ttl_seconds
        await redis.zadd(settings.temp_redis_zset_key, {file_path: expiry_timestamp})
        logger.debug(f"Registered temp file in Redis: {file_path} (expires at {expiry_timestamp})")
    except Exception as e:
        logger.warning(f"Failed to register temp file in Redis: {e}")


async def remove_temp_registration(redis: Optional["aioredis.Redis"], file_path: str) -> None:
    """Remove temp file registration from Redis (call when download starts)
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file
    """
    if not redis:
        return
    
    try:
    await redis.zrem(settings.temp_redis_zset_key, file_path)
        logger.debug(f"Removed temp file registration from Redis: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temp file from Redis: {e}")


async def check_file_exists_in_redis(redis: Optional["aioredis.Redis"], file_path: str) -> bool:
    """Check if temp file is still registered in Redis before I/O operation
    
    Args:
        redis: Redis client (None if Redis not configured)
        file_path: Absolute path to temp file
        
    Returns:
        True if file exists in Redis, False if not found or Redis unavailable
    """
    if not redis:
        # Fallback to filesystem check
        return await aiofiles.os.path.exists(file_path)
    
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
        # Fallback to filesystem check
        return await aiofiles.os.path.exists(file_path)


async def redis_cleanup_once(redis: "aioredis.Redis", batch_size: int = 100) -> int:
    """Execute one cleanup cycle: find expired files, delete them from Redis and filesystem
    
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
        
        # Delete files from filesystem
        deleted_count = 0
        for file_path_bytes in expired_files:
            file_path = file_path_bytes.decode("utf-8") if isinstance(file_path_bytes, bytes) else file_path_bytes
            try:
                if await aiofiles.os.path.exists(file_path):
                    await aiofiles.os.remove(file_path)
                    logger.info(f"Deleted expired temp file: {file_path}")
                    deleted_count += 1
                else:
                    logger.debug(f"Temp file already deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
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
    """
    if not settings.enable_temp_file_system:
        logger.debug("Filesystem temp cleanup disabled")
        return
    
    try:
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
    """Handles writing audio chunks to a temp file with Redis lifecycle management"""

    def __init__(self, format: str, redis: Optional["aioredis.Redis"] = None):
        """Initialize temp file writer

        Args:
            format: Audio format extension (mp3, wav, etc)
            redis: Optional Redis client for lifecycle management
        """
        self.format = format
        self.redis = redis
        self.temp_file = None
        self._finalized = False
        self._write_error = False  # Flag to track if we've had a write error

    async def __aenter__(self):
        """Async context manager entry"""
        try:
            # Clean up old files first (only if filesystem cleanup is enabled)
            if settings.enable_temp_file_system:
                await cleanup_temp_files()

            # Create temp file with UUID-based name
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            
            # Generate UUID-based filename
            file_uuid = uuid.uuid4().hex
            filename = f"{file_uuid}.{self.format}"
            self.temp_path = os.path.join(settings.temp_file_dir, filename)
            
            # Create and open file
            self.temp_file = await aiofiles.open(self.temp_path, mode="wb")

            # Generate download path immediately
            self.download_path = f"/download/{os.path.basename(self.temp_path)}"
            
            # Register in Redis if available
            if self.redis:
                await register_temp_file(self.redis, self.temp_path, settings.temp_file_ttl_seconds)
            
            logger.debug(f"Created temp file: {self.temp_path}")
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to create temp file: {e}")
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
        if self._write_error or not self.temp_file:
            return

        try:
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
        if self._write_error or not self.temp_file:
            self._finalized = True
            return self.download_path

        try:
            await self.temp_file.close()
            self._finalized = True
        except Exception as e:
            # Handle permission issues or other errors gracefully
            logger.error(f"Failed to finalize temp file: {e}")
            self._write_error = True
            self._finalized = True

        return self.download_path
