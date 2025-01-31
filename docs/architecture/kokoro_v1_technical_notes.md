# Kokoro v1.0 Technical Integration Notes

## Core Components

1. KModel Class
- Main model class with unified interface
- Handles both weights and inference
- Language-blind design (phoneme focused)
- No external language processing

2. Key Architecture Changes
- Uses CustomAlbert instead of PLBert
- New ProsodyPredictor implementation
- Different phoneme handling approach
- Built-in vocab management

## Integration Points

1. Model Loading
```python
# v1.0 approach
model = KModel(config_path, model_path)
# vs our current
model = await build_model(path, device)
```

2. Forward Pass Differences
```python
# v1.0
audio = model(phonemes, ref_s, speed=1.0)
# vs our current
audio = model.decoder(asr, F0_pred, N_pred, ref_s)
```

3. Key Dependencies
- transformers (for AlbertConfig)
- torch
- No external phoneme processing

## Configuration Changes

1. v1.0 Config Structure
```json
{
  "vocab": {...},  # Built-in phoneme mapping
  "n_token": X,
  "plbert": {...},  # Albert config
  "hidden_dim": X,
  "style_dim": X,
  "istftnet": {...}
}
```

2. Voice Management
- No HF downloads
- Local voice file management
- Simpler voice structure

## Implementation Strategy

1. Core Changes
- Keep our streaming infrastructure
- Adapt to new model interface
- Maintain our voice management

2. Key Adaptations Needed
- Wrap KModel in our build system
- Handle phoneme mapping internally
- Adapt to new prosody prediction

3. Compatibility Layer
```python
class V1ModelWrapper:
    def __init__(self, kmodel):
        self.model = kmodel
        
    async def forward(self, phonemes, ref_s):
        # Adapt v1.0 interface to our system
        return self.model(phonemes, ref_s)
```

## Technical Considerations

1. Memory Usage
- Models ~few hundred MB
- Voices ~few hundred KB
- No need for complex memory management

2. Performance
- Similar inference speed expected
- No major architectural bottlenecks
- Keep existing streaming optimizations

3. Integration Points
- Model loading/initialization
- Voice file management
- Inference pipeline
- Streaming output

## Migration Notes

1. Key Files to Port
- model.py -> v1_0/models.py
- istftnet.py -> v1_0/istftnet.py
- Add albert.py for CustomAlbert

2. Config Updates
- Add version selection
- Keep config structure similar
- Add v1.0 specific params

3. Testing Focus
- Basic inference
- Voice compatibility
- Streaming performance
- Version switching