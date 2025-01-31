"""Tests for TTSService v2 with version support."""

import numpy as np
import pytest
from loguru import logger

from ..services.tts_service_v2 import TTSService


@pytest.fixture
async def tts_service():
    """Fixture for TTSService instance."""
    service = await TTSService.create()
    yield service


@pytest.mark.asyncio
async def test_service_initialization(tts_service):
    """Test TTSService initialization."""
    assert tts_service is not None
    assert tts_service.version_manager is not None
    assert tts_service._voice_manager is not None


@pytest.mark.asyncio
async def test_version_selection(tts_service):
    """Test version selection in TTSService."""
    # Default version should be v0.19
    assert tts_service.current_version == "v0.19"
    
    # Change version
    tts_service.set_version("v1.0")
    assert tts_service.current_version == "v1.0"
    
    # List available versions
    versions = tts_service.available_versions
    assert "v0.19" in versions
    assert "v1.0" in versions


@pytest.mark.asyncio
async def test_audio_generation_v0_19(tts_service):
    """Test audio generation with v0.19."""
    text = "Hello, world!"
    voice = "af_bella"  # Use a known test voice
    
    # Set version explicitly
    tts_service.set_version("v0.19")
    
    # Generate audio
    audio, processing_time = await tts_service.generate_audio(
        text=text,
        voice=voice,
        speed=1.0,
        version="v0.19"
    )
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert processing_time > 0


@pytest.mark.asyncio
async def test_audio_generation_v1_0(tts_service):
    """Test audio generation with v1.0."""
    text = "Hello, world!"
    voice = "af_bella"  # Use a known test voice
    
    # Set version explicitly
    tts_service.set_version("v1.0")
    
    # Generate audio
    audio, processing_time = await tts_service.generate_audio(
        text=text,
        voice=voice,
        speed=1.0,
        version="v1.0"
    )
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert processing_time > 0


@pytest.mark.asyncio
async def test_streaming_v0_19(tts_service):
    """Test audio streaming with v0.19."""
    text = "Hello, world!"
    voice = "af_bella"
    chunks = []
    
    async for chunk in tts_service.generate_audio_stream(
        text=text,
        voice=voice,
        speed=1.0,
        version="v0.19",
        output_format="wav"
    ):
        assert chunk is not None
        assert len(chunk) > 0
        chunks.append(chunk)
    
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_streaming_v1_0(tts_service):
    """Test audio streaming with v1.0."""
    text = "Hello, world!"
    voice = "af_bella"
    chunks = []
    
    async for chunk in tts_service.generate_audio_stream(
        text=text,
        voice=voice,
        speed=1.0,
        version="v1.0",
        output_format="wav"
    ):
        assert chunk is not None
        assert len(chunk) > 0
        chunks.append(chunk)
    
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_voice_compatibility(tts_service):
    """Test voice compatibility across versions."""
    # List voices
    voices = await tts_service.list_voices()
    assert len(voices) > 0
    
    # Test a voice with both versions
    test_voice = "af_bella"
    assert test_voice in voices
    
    # Test with v0.19
    audio_v0, _ = await tts_service.generate_audio(
        text="Test",
        voice=test_voice,
        version="v0.19"
    )
    assert isinstance(audio_v0, np.ndarray)
    assert len(audio_v0) > 0
    
    # Test with v1.0
    audio_v1, _ = await tts_service.generate_audio(
        text="Test",
        voice=test_voice,
        version="v1.0"
    )
    assert isinstance(audio_v1, np.ndarray)
    assert len(audio_v1) > 0


@pytest.mark.asyncio
async def test_invalid_version(tts_service):
    """Test handling of invalid version."""
    with pytest.raises(ValueError):
        await tts_service.generate_audio(
            text="Test",
            voice="af_bella",
            version="invalid_version"
        )


@pytest.mark.asyncio
async def test_invalid_voice(tts_service):
    """Test handling of invalid voice."""
    with pytest.raises(ValueError):
        await tts_service.generate_audio(
            text="Test",
            voice="invalid_voice",
            version="v1.0"
        )


@pytest.mark.asyncio
async def test_empty_text(tts_service):
    """Test handling of empty text."""
    with pytest.raises(ValueError):
        await tts_service.generate_audio(
            text="",
            voice="af_bella",
            version="v1.0"
        )


@pytest.mark.asyncio
async def test_voice_combination(tts_service):
    """Test voice combination with different versions."""
    voices = ["af_bella", "af_nicole"]
    
    # Combine voices
    combined_voice = await tts_service.combine_voices(voices)
    assert combined_voice is not None
    
    # Test combined voice with both versions
    for version in ["v0.19", "v1.0"]:
        audio, _ = await tts_service.generate_audio(
            text="Test combined voice",
            voice=combined_voice,
            version=version
        )
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0