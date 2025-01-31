# Kokoro v1.0 Wrapper Technical Specification

## Overview

This document details the technical implementation of the KokoroV1Wrapper class that integrates the Kokoro v1.0 KModel/KPipeline architecture with our existing system.

## Class Implementation

```python
from pathlib import Path
from kokoro import KModel, KPipeline

class KokoroV1Wrapper:
    """Wrapper for Kokoro v1.0 KModel/KPipeline integration.
    
    This wrapper manages:
    1. Model initialization and weight loading
    2. Pipeline creation and caching per language
    3. Streaming audio generation
    """
    
    def __init__(self, config_path: str, model_path: str):
        """Initialize KModel with config and weights.
        
        Args:
            config_path: Path to config.json in builds/v1_0/
            model_path: Path to model weights in models/v1_0/
        """
        self.model = KModel()  # Will load config and weights
        self.pipelines = {}    # lang_code -> KPipeline cache
        
    def get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create a KPipeline for the given language code.
        
        Args:
            lang_code: Language code for phoneme processing
            
        Returns:
            KPipeline instance for the language
        """
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                model=self.model
            )
        return self.pipelines[lang_code]
        
    async def forward(self, text: str, voice: str, lang_code: str):
        """Generate audio using the appropriate pipeline.
        
        Args:
            text: Input text to synthesize
            voice: Voice ID to use
            lang_code: Language code for phoneme processing
            
        Yields:
            Audio chunks as torch.FloatTensor
        """
        pipeline = self.get_pipeline(lang_code)
        generator = pipeline(text, voice=voice)
        for gs, ps, audio in generator:
            yield audio

class ModelManager:
    """Manages multiple model versions and their initialization."""
    
    def __init__(self):
        self.models = {}
        
    async def get_model(self, version: str):
        """Get or initialize a model for the specified version.
        
        Args:
            version: Model version ("v0.19" or "v1.0")
            
        Returns:
            Initialized model instance
        """
        if version not in self.models:
            if version == "v0.19":
                from ..builds.v0_19.models import build_model
                self.models[version] = await build_model()
            elif version == "v1.0":
                from ..builds.v1_0.wrapper import KokoroV1Wrapper
                
                # Config in builds directory
                config_path = Path(__file__).parent / "builds/v1_0/config.json"
                
                # Model weights in models directory
                model_path = Path(__file__).parent / "models/v1_0/kokoro-v1_0.pth"
                
                self.models[version] = KokoroV1Wrapper(
                    config_path=str(config_path),
                    model_path=str(model_path)
                )
        return self.models[version]
```

## Key Design Points

1. Model Management
   - KModel handles weights and inference
   - Config and weights loaded from separate directories
   - Language-blind design (phoneme focused)

2. Pipeline Caching
   - One KPipeline per language code
   - Pipelines created on demand and cached
   - Reuses single KModel instance

3. Streaming Integration
   - Maintains compatibility with existing streaming system
   - Yields audio chunks progressively
   - Handles both quiet and loud pipeline modes

4. Version Control
   - Clear separation between v0.19 and v1.0
   - Version-specific model initialization
   - Shared model manager interface

## Usage Example

```python
# Initialize model manager
manager = ModelManager()

# Get v1.0 model
model = await manager.get_model("v1.0")

# Generate audio
async for audio in model.forward(
    text="Hello world",
    voice="af_bella",
    lang_code="en"
):
    # Process audio chunk
    process_audio(audio)
```

## Error Handling

1. File Access
   - Verify config.json exists in builds/v1_0/
   - Verify model weights exist in models/v1_0/
   - Handle missing or corrupt files

2. Pipeline Creation
   - Validate language codes
   - Handle initialization failures
   - Clean up failed pipeline instances

3. Voice Loading
   - Verify voice file existence
   - Handle voice format compatibility
   - Manage voice loading failures

## Testing Strategy

1. Unit Tests
   - Model initialization
   - Pipeline creation and caching
   - Audio generation
   - Error handling

2. Integration Tests
   - End-to-end audio generation
   - Streaming performance
   - Memory usage
   - Multi-language support

3. Performance Tests
   - Pipeline creation overhead
   - Memory usage patterns
   - Streaming latency
   - Voice loading speed