"""
Kokoro v1.0 model build system integration.
"""
from pathlib import Path
from typing import Optional

from .wrapper import build_model as build_v1_model


async def build_model(path: Optional[str] = None, device: str = "cuda"):
    """Build a Kokoro v1.0 model instance.
    
    This function maintains compatibility with the v0.19 build_model interface
    while using the new KModel/KPipeline architecture internally.
    
    Args:
        path: Optional path to model weights. If None, uses default location.
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Initialized model instance
    """
    if path is None:
        # Use default path in models/v1_0
        path = str(Path(__file__).parent.parent.parent / "models/v1_0/kokoro-v1_0.pth")
    
    # Config is always in builds/v1_0
    config_path = str(Path(__file__).parent / "config.json")
    
    return await build_v1_model(
        config_path=config_path,
        model_path=path
    )