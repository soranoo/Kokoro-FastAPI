"""Version-aware model management."""

import asyncio
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from loguru import logger

from ..builds.v0_19.models import build_model as build_v0_19
from ..builds.v1_0.models import build_model as build_v1
from ..core.config import settings


# Global singleton instance and lock for thread-safe initialization
_manager_instance = None
_manager_lock = asyncio.Lock()


class VersionManager:
    """Manages different versions of Kokoro models."""

    def __init__(self):
        """Initialize version manager."""
        self.models: Dict[str, Union[dict, object]] = {}
        self._version_locks: Dict[str, asyncio.Lock] = {
            "v0.19": asyncio.Lock(),
            "v1.0": asyncio.Lock()
        }
        self._current_version = "v1.0"  # Default to v1.0 with af_bella voice

    async def get_model(self, version: Optional[str] = None) -> object:
        """Get model for specified version.
        
        Args:
            version: Model version ("v0.19" or "v1.0"). Uses default if None.
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If version is invalid
            RuntimeError: If model loading fails
        """
        if version is None:
            version = self._current_version

        if version not in self._version_locks:
            raise ValueError(
                f"Invalid version: {version}. "
                f"Available versions: {', '.join(self._version_locks.keys())}"
            )

        # Fast path - return existing model
        if version in self.models and self.models[version] is not None:
            return self.models[version]

        # Slow path - load model with lock
        async with self._version_locks[version]:
            # Double-check pattern
            if version not in self.models or self.models[version] is None:
                try:
                    if version == "v0.19":
                        # Use existing model path logic for v0.19
                        from ..core.model_config import model_config
                        model_file = (model_config.onnx_model_file 
                                    if settings.use_onnx 
                                    else model_config.pytorch_model_file)
                        from ..core.paths import get_model_path
                        model_path = await get_model_path(model_file)
                        self.models[version] = await build_v0_19(
                            path=model_path,
                            device="cuda" if settings.use_gpu else "cpu"
                        )
                    else:  # v1.0
                        # Use paths module for v1.0 model loading
                        from ..core.paths import get_model_path
                        model_path = await get_model_path("kokoro-v1_0.pth")
                        self.models[version] = await build_v1(
                            path=model_path,
                            device="cuda" if settings.use_gpu else "cpu"
                        )
                    logger.info(f"Loaded {version} model")
                except Exception as e:
                    logger.error(f"Failed to load {version} model: {e}")
                    raise RuntimeError(f"Failed to load {version} model: {e}")

            return self.models[version]

    def set_default_version(self, version: str) -> None:
        """Set default model version.
        
        Args:
            version: Version to set as default ("v0.19" or "v1.0")
            
        Raises:
            ValueError: If version is invalid
        """
        if version not in self._version_locks:
            raise ValueError(
                f"Invalid version: {version}. "
                f"Available versions: {', '.join(self._version_locks.keys())}"
            )
        self._current_version = version
        logger.info(f"Set default version to {version}")

    @property
    def current_version(self) -> str:
        """Get current default version."""
        return self._current_version

    @property
    def available_versions(self) -> list[str]:
        """Get list of available versions."""
        return list(self._version_locks.keys())

    def unload_all(self) -> None:
        """Unload all model versions."""
        self.models.clear()
        logger.info("Unloaded all model versions")


async def get_version_manager() -> VersionManager:
    """Get global version manager instance.
    
    Returns:
        VersionManager instance
        
    Thread Safety:
        This function is thread-safe
    """
    global _manager_instance
    
    # Fast path - return existing instance
    if _manager_instance is not None:
        return _manager_instance
        
    # Slow path - create new instance with lock
    async with _manager_lock:
        # Double-check pattern
        if _manager_instance is None:
            _manager_instance = VersionManager()
        return _manager_instance