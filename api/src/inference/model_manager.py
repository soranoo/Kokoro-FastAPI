"""Kokoro V1 model management."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelInstance:
    """Individual model instance with its own CUDA stream."""

    def __init__(self, instance_id: int):
        """Initialize model instance."""
        self.instance_id = instance_id
        self._backend: Optional[KokoroV1] = None
        self._device: Optional[str] = None
        self._stream: Optional[torch.cuda.Stream] = None if not settings.use_gpu else torch.cuda.Stream()
        self._in_use = False
        self._last_used = 0.0

    @property
    def is_available(self) -> bool:
        """Check if instance is available."""
        return not self._in_use

    async def initialize(self) -> None:
        """Initialize model instance."""
        try:
            self._device = "cuda" if settings.use_gpu else "cpu"
            logger.info(f"Initializing Kokoro V1 instance {self.instance_id} on {self._device}")
            self._backend = KokoroV1()
            if self._stream:
                self._backend.set_stream(self._stream)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1 instance {self.instance_id}: {e}")

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend."""
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model for instance {self.instance_id}: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend."""
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed for instance {self.instance_id}: {e}")

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None


class ModelPool:
    """Pool of model instances."""

    def __init__(self, max_instances: int):
        """Initialize model pool."""
        self.max_instances = max_instances
        self._instances: List[ModelInstance] = []
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=model_config.pytorch_gpu.max_queue_size)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize model pool."""
        async with self._lock:
            for i in range(self.max_instances):
                instance = ModelInstance(i)
                await instance.initialize()
                self._instances.append(instance)

    async def get_instance(self) -> ModelInstance:
        """Get available model instance or wait for one."""
        while True:
            # Try to find an available instance
            for instance in self._instances:
                if instance.is_available:
                    instance._in_use = True
                    instance._last_used = time.time()
                    return instance

            # If no instance is available, wait in queue
            try:
                await self._request_queue.put(asyncio.current_task())
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.QueueFull:
                raise RuntimeError("Request queue is full")

    async def release_instance(self, instance: ModelInstance) -> None:
        """Release model instance back to pool."""
        instance._in_use = False
        # Process next request in queue if any
        if not self._request_queue.empty():
            waiting_task = await self._request_queue.get()
            if not waiting_task.done():
                waiting_task.set_result(None)


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager."""
        self._config = config or model_config
        self._pool: Optional[ModelPool] = None
        self._chunk_semaphore = asyncio.Semaphore(self._config.pytorch_gpu.chunk_semaphore_limit)

    async def initialize(self) -> None:
        """Initialize model pool."""
        if not self._pool:
            self._pool = ModelPool(self._config.pytorch_gpu.max_concurrent_models)
            await self._pool.initialize()

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model pool.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)
        """
        try:
            # Initialize pool
            await self.initialize()

            # Load model on all instances
            model_path = self._config.pytorch_kokoro_v1_file
            for instance in self._pool._instances:
                await instance.load_model(model_path)

            # Warm up first instance
            instance = self._pool._instances[0]
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text
                warmup_text = "Warmup text for initialization."
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in instance.generate(warmup_text, (voice_name, voice_path)):
                    pass

            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            return "cuda" if settings.use_gpu else "cpu", "kokoro_v1", len(voices)

        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using model pool."""
        if not self._pool:
            raise RuntimeError("Model pool not initialized")

        # Get available instance
        instance = await self._pool.get_instance()
        try:
            async with self._chunk_semaphore:
                async for chunk in instance.generate(*args, **kwargs):
                    yield chunk
        finally:
            # Release instance back to pool
            await self._pool.release_instance(instance)

    def unload_all(self) -> None:
        """Unload all models and free resources."""
        if self._pool:
            for instance in self._pool._instances:
                instance.unload()
            self._pool = None


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
