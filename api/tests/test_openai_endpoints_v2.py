"""Tests for OpenAI-compatible v2 endpoints."""

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from ..main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_versions(client):
    """Test version listing endpoint."""
    response = client.get("/v2/audio/versions")
    assert response.status_code == 200
    data = response.json()
    assert "versions" in data
    assert "current" in data
    assert "v0.19" in data["versions"]
    assert "v1.0" in data["versions"]


def test_set_version(client):
    """Test version setting endpoint."""
    # Set to v1.0
    response = client.post("/v2/audio/version", json="v1.0")
    assert response.status_code == 200
    data = response.json()
    assert data["current"] == "v1.0"
    
    # Set back to v0.19
    response = client.post("/v2/audio/version", json="v0.19")
    assert response.status_code == 200
    data = response.json()
    assert data["current"] == "v0.19"
    
    # Test invalid version
    response = client.post("/v2/audio/version", json="invalid_version")
    assert response.status_code == 400


def test_list_voices(client):
    """Test voice listing endpoint."""
    response = client.get("/v2/audio/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert len(data["voices"]) > 0


def test_combine_voices(client):
    """Test voice combination endpoint."""
    # Test with string input
    response = client.post("/v2/audio/voices/combine", json="af_bella+af_nicole")
    assert response.status_code == 200
    data = response.json()
    assert "voice" in data
    assert "voices" in data
    
    # Test with list input
    response = client.post("/v2/audio/voices/combine", json=["af_bella", "af_nicole"])
    assert response.status_code == 200
    data = response.json()
    assert "voice" in data
    assert "voices" in data


def test_speech_generation_v0_19(client):
    """Test speech generation with v0.19."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
        "version": "v0.19"
    }
    
    response = client.post("/v2/audio/speech", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0


def test_speech_generation_v1_0(client):
    """Test speech generation with v1.0."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
        "version": "v1.0"
    }
    
    response = client.post("/v2/audio/speech", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0


def test_streaming_speech_v0_19(client):
    """Test streaming speech generation with v0.19."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": True,
        "version": "v0.19"
    }
    
    with client.stream("POST", "/v2/audio/speech", json=request_data) as response:
        assert response.status_code == 200
        content = b""
        for chunk in response.iter_bytes():
            assert len(chunk) > 0
            content += chunk
        assert len(content) > 0


def test_streaming_speech_v1_0(client):
    """Test streaming speech generation with v1.0."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": True,
        "version": "v1.0"
    }
    
    with client.stream("POST", "/v2/audio/speech", json=request_data) as response:
        assert response.status_code == 200
        content = b""
        for chunk in response.iter_bytes():
            assert len(chunk) > 0
            content += chunk
        assert len(content) > 0


def test_invalid_model(client):
    """Test invalid model handling."""
    request_data = {
        "model": "invalid-model",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "version": "v1.0"
    }
    
    response = client.post("/v2/audio/speech", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "invalid_model"


def test_invalid_voice(client):
    """Test invalid voice handling."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "invalid_voice",
        "response_format": "wav",
        "version": "v1.0"
    }
    
    response = client.post("/v2/audio/speech", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "validation_error"


def test_invalid_version(client):
    """Test invalid version handling."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "version": "invalid_version"
    }
    
    response = client.post("/v2/audio/speech", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "validation_error"


def test_download_link(client):
    """Test download link functionality."""
    request_data = {
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": True,
        "return_download_link": True,
        "version": "v1.0"
    }
    
    with client.stream("POST", "/v2/audio/speech", json=request_data) as response:
        assert response.status_code == 200
        assert "X-Download-Path" in response.headers
        download_path = response.headers["X-Download-Path"]
        
        # Try downloading the file
        download_response = client.get(f"/download/{download_path}")
        assert download_response.status_code == 200
        assert len(download_response.content) > 0