# Kokoro v1.0 Implementation Checklist

## 1. Directory Setup
- [ ] Create api/src/models/v1_0/
- [ ] Create api/src/builds/v0_19/
- [ ] Create api/src/builds/v1_0/
- [ ] Create api/src/voices/v0_19/
- [ ] Create api/src/voices/v1_0/

## 2. Move Current Implementation to v0.19
- [ ] Move api/src/builds/config.json -> api/src/builds/v0_19/config.json
- [ ] Move api/src/builds/models.py -> api/src/builds/v0_19/models.py
- [ ] Move api/src/builds/istftnet.py -> api/src/builds/v0_19/istftnet.py
- [ ] Move api/src/builds/plbert.py -> api/src/builds/v0_19/plbert.py
- [ ] Move current voices to api/src/voices/v0_19/

## 3. Copy v1.0 Files from Kokoro-82M
- [ ] Copy kokoro-v1_0.pth -> api/src/models/v1_0/kokoro-v1_0.pth
- [ ] Copy config.json -> api/src/builds/v1_0/config.json
- [ ] Copy voices/*.pt -> api/src/voices/v1_0/

## 4. Create v1.0 Implementation
- [ ] Create api/src/builds/v1_0/wrapper.py
- [ ] Create api/src/builds/v1_0/models.py
- [ ] Update imports in moved files
- [ ] Install kokoro package dependencies

## 5. Update Model Manager
- [ ] Update model manager to support both versions
- [ ] Add version-specific model loading
- [ ] Add version selection parameter to API endpoints

## 6. Testing
- [ ] Test v0.19 functionality still works
- [ ] Test v1.0 model loading
- [ ] Test voice loading for both versions
- [ ] Test streaming functionality
- [ ] Test version switching

## 7. Documentation
- [ ] Update API documentation with version parameter
- [ ] Document voice compatibility
- [ ] Add migration guide for users

## Notes
- Model weights go in api/src/models/v1_0/
- Build configs and code go in api/src/builds/v1_0/
- Keep voice files local, no HF downloads
- Test each step before proceeding to next