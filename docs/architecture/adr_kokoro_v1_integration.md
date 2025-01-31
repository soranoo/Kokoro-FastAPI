# Architectural Decision Record: Kokoro v1.0 Integration

## Context

We are integrating Kokoro v1.0 while maintaining backward compatibility with v0.19. The v1.0 release introduces significant architectural changes including a new KModel/KPipeline design, language-blind model architecture, and built-in vocab management.

## Decision

We will implement a hybrid architecture that:

1. Maintains existing streaming infrastructure
2. Supports both v0.19 and v1.0 models
3. Adapts the new KModel/KPipeline interface to our system

### Key Components

#### 1. Version-Specific Model Builds
```
api/src/builds/
├── v0_19/          # Current implementation
└── v1_0/           # New implementation using KModel
```

#### 2. Model Manager Interface
```python
class ModelManager:
    def __init__(self):
        self.models = {}  # version -> model
        
    async def get_model(self, version: str):
        if version not in self.models:
            if version == "v0.19":
                from ..builds.v0_19.models import build_model
            elif version == "v1.0":
                from ..builds.v1_0.models import build_model
            self.models[version] = await build_model()
        return self.models[version]
```

#### 3. Voice Management
```
api/src/voices/
├── v0_19/
└── v1_0/
```

### Integration Strategy

1. Model Integration
   - Wrap KModel in our build system
   - Adapt to new forward pass interface
   - Handle phoneme mapping internally

2. Pipeline Integration
   ```python
   class V1ModelWrapper:
       def __init__(self, kmodel):
           self.model = kmodel
           self.pipeline = KPipeline(model=kmodel)
           
       async def forward(self, text, voice):
           # Adapt v1.0 interface to our streaming system
           generator = self.pipeline(text, voice=voice)
           for gs, ps, audio in generator:
               yield audio
   ```

3. API Layer
   - Add version parameter to endpoints
   - Default to v1.0 if not specified
   - Maintain backward compatibility

## Consequences

### Positive
- Clean separation between v0.19 and v1.0 implementations
- Minimal changes to existing streaming infrastructure
- Simple version switching mechanism
- Local voice management maintained

### Negative
- Some code duplication between versions
- Additional wrapper layer for v1.0
- Need to maintain two parallel implementations

### Neutral
- Similar memory footprint (models ~few hundred MB)
- Comparable inference speed expected
- No major architectural bottlenecks

## Implementation Plan

1. Directory Structure Setup
   - Create version-specific directories
   - Move current implementation to v0_19/

2. V1.0 Integration
   - Implement KModel wrapper
   - Add version-aware model manager
   - Setup voice directory structure

3. Testing Focus
   - Basic inference for both versions
   - Voice compatibility
   - Streaming performance
   - Version switching
   - API endpoint compatibility

## Migration Path

1. Initial Release
   - Both versions available
   - v0.19 as default

2. Transition Period
   - v1.0 as default
   - v0.19 still available

3. Future
   - Consider deprecation timeline for v0.19
   - Document migration path for users