# Kokoro Version Support Architecture

## Overview

Simple architecture for supporting both Kokoro v0.19 and v1.0 models, allowing version selection via API.

## Directory Structure

```
api/src/builds/
├── v0_19/                # Current implementation
│   ├── config.json      
│   ├── models.py        
│   ├── istftnet.py      
│   └── plbert.py        
└── v1_0/                # New v1.0 implementation
    ├── config.json     
    ├── models.py       
    ├── istftnet.py    
    └── albert.py      
```

## Implementation Plan

1. Move Current Implementation
   - Relocate existing files to v0_19/
   - Update imports

2. Add v1.0 Implementation
   - Copy reference implementation
   - Adapt to our structure
   - Keep voice management local

3. Model Manager Updates
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

4. API Integration
   - Add version parameter to endpoints
   - Default to v1.0 if not specified

## Voice Management
- Simple directory structure:
  ```
  api/src/voices/
  ├── v0_19/
  └── v1_0/
  ```
- Keep voice files local, no HF downloads

## Testing
- Basic functionality tests for each version
- Version switching tests
- Voice compatibility tests

No need to over-optimize - models and voices are small enough to keep things simple.