"""Tests for version manager functionality."""

import pytest
from loguru import logger

from ..inference.version_manager import get_version_manager


@pytest.mark.asyncio
async def test_version_manager_initialization():
    """Test version manager initialization."""
    manager = await get_version_manager()
    assert manager is not None
    assert manager.current_version == "v0.19"  # Default version
    assert set(manager.available_versions) == {"v0.19", "v1.0"}


@pytest.mark.asyncio
async def test_version_switching():
    """Test switching between model versions."""
    manager = await get_version_manager()
    
    # Switch to v1.0
    manager.set_default_version("v1.0")
    assert manager.current_version == "v1.0"
    
    # Switch back to v0.19
    manager.set_default_version("v0.19")
    assert manager.current_version == "v0.19"
    
    # Test invalid version
    with pytest.raises(ValueError):
        manager.set_default_version("invalid_version")


@pytest.mark.asyncio
async def test_model_loading():
    """Test loading models for different versions."""
    manager = await get_version_manager()
    
    # Load v0.19 model
    v0_model = await manager.get_model("v0.19")
    assert v0_model is not None
    
    # Load v1.0 model
    v1_model = await manager.get_model("v1.0")
    assert v1_model is not None
    
    # Models should be cached
    v0_model_cached = await manager.get_model("v0.19")
    assert v0_model_cached is v0_model
    
    v1_model_cached = await manager.get_model("v1.0")
    assert v1_model_cached is v1_model


@pytest.mark.asyncio
async def test_model_unloading():
    """Test unloading all models."""
    manager = await get_version_manager()
    
    # Load both models
    await manager.get_model("v0.19")
    await manager.get_model("v1.0")
    
    # Unload all
    manager.unload_all()
    assert not manager.models
    
    # Models should reload when requested
    v0_model = await manager.get_model("v0.19")
    assert v0_model is not None
    
    v1_model = await manager.get_model("v1.0")
    assert v1_model is not None


@pytest.mark.asyncio
async def test_invalid_model_request():
    """Test requesting invalid model version."""
    manager = await get_version_manager()
    
    with pytest.raises(ValueError):
        await manager.get_model("invalid_version")


@pytest.mark.asyncio
async def test_default_version_model():
    """Test getting model with default version."""
    manager = await get_version_manager()
    
    # Get model without specifying version
    default_model = await manager.get_model()
    assert default_model is not None
    
    # Should match explicitly requesting v0.19
    v0_model = await manager.get_model("v0.19")
    assert default_model is v0_model
    
    # Change default and verify
    manager.set_default_version("v1.0")
    new_default_model = await manager.get_model()
    v1_model = await manager.get_model("v1.0")
    assert new_default_model is v1_model