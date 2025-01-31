# Kokoro v1.0 File Migration Plan

## Source Files (Kokoro-82M)

```
Kokoro-82M/
├── config.json          # Model configuration
├── kokoro-v1_0.pth     # Model weights
└── voices/             # Voice files
    ├── af_alloy.pt
    ├── af_aoede.pt
    ├── af_bella.pt
    └── ...
```

## Target Structure

```
api/src/builds/
├── v0_19/                # Current implementation
│   ├── config.json      # Move current config.json here
│   ├── models.py        # Move current models.py here
│   ├── istftnet.py      # Move current istftnet.py here
│   └── plbert.py        # Move current plbert.py here
└── v1_0/                # New implementation
    ├── config.json      # From Kokoro-82M/config.json
    ├── models.py        # To be created - build system integration
    └── wrapper.py       # To be created - KModel/KPipeline wrapper

api/src/models/
└── v1_0/                # Model weights directory
    └── kokoro-v1_0.pth  # From Kokoro-82M/kokoro-v1_0.pth

api/src/voices/
├── v0_19/               # Current voices
│   ├── af_bella.pt
│   ├── af_nicole.pt
│   └── ...
└── v1_0/               # From Kokoro-82M/voices/
    ├── af_alloy.pt
    ├── af_aoede.pt
    └── ...
```

## Migration Steps

1. Create Directory Structure
   - Create api/src/builds/v0_19/
   - Create api/src/builds/v1_0/
   - Create api/src/models/v1_0/
   - Create api/src/voices/v0_19/
   - Create api/src/voices/v1_0/

2. Move Current Implementation to v0.19
   - Move api/src/builds/config.json -> api/src/builds/v0_19/config.json
   - Move api/src/builds/models.py -> api/src/builds/v0_19/models.py
   - Move api/src/builds/istftnet.py -> api/src/builds/v0_19/istftnet.py
   - Move api/src/builds/plbert.py -> api/src/builds/v0_19/plbert.py
   - Move api/src/voices/*.pt -> api/src/voices/v0_19/

3. Copy v1.0 Files
   - Copy Kokoro-82M/config.json -> api/src/builds/v1_0/config.json
   - Copy Kokoro-82M/kokoro-v1_0.pth -> api/src/models/v1_0/kokoro-v1_0.pth
   - Copy Kokoro-82M/voices/*.pt -> api/src/voices/v1_0/

4. Create New Implementation Files
   - Create api/src/builds/v1_0/wrapper.py for KModel integration
   - Create api/src/builds/v1_0/models.py for build system integration

## Implementation Notes

1. Voice Management
   - Keep voice files local, no HF downloads
   - Maintain compatibility with both versions
   - Consider voice file format differences

2. Model Integration
   - Use kokoro package for v1.0 model loading
   - Model weights accessed from api/src/models/v1_0/
   - Adapt to our streaming infrastructure
   - Handle version-specific configurations

3. Testing Considerations
   - Verify file permissions after moves
   - Test voice loading from both directories
   - Ensure backward compatibility
   - Validate streaming performance

4. Code Updates
   - Update model loading paths in wrapper.py to point to api/src/models/v1_0/
   - Maintain separation between model weights and build configuration
   - Ensure proper error handling for missing files