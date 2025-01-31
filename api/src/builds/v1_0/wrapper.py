from pathlib import Path
from typing import AsyncGenerator, List

import numpy as np
import torch
from kokoro import KModel
from loguru import logger


class KokoroV1Wrapper:
    """Wrapper for Kokoro v1.0 KModel integration.
    
    This wrapper provides a token-based interface compatible with the TTS service,
    while internally using the Kokoro KModel for direct audio generation. It handles:
    
    - Token-to-phoneme conversion using the model's vocab
    - Voice tensor management
    - Audio generation with speed control
    """
    
    def __init__(self, config_path: str, model_path: str):
        """Initialize KModel with config and weights.
        
        Args:
            config_path: Path to config.json in builds/v1_0/
            model_path: Path to model weights in models/v1_0/
        """
        self.model = KModel(config=config_path, model=model_path)
        self.vocab = self.model.vocab  # Get vocab from model for token decoding
        
    async def forward(self, tokens: List[int], voice_tensor: torch.Tensor, speed: float = 1.0) -> AsyncGenerator[torch.FloatTensor, None]:
        """Generate audio using KModel's forward pass.
        
        Args:
            tokens: Input token sequence to convert to phonemes
            voice_tensor: Voice embedding tensor (ref_s) containing style information
            speed: Speed multiplier for audio generation
            
        Yields:
            Single audio tensor as torch.FloatTensor
            
        Raises:
            RuntimeError: If token-to-phoneme conversion or audio generation fails
        """
        try:
            # Convert tokens back to phonemes using vocab
            phonemes = []
            for token in tokens:
                for p, idx in self.vocab.items():
                    if idx == token:
                        phonemes.append(p)
                        break
            text = ''.join(phonemes)
            logger.debug(f"Decoded tokens to text: '{text[:100]}...'")
            
            # Validate and reshape voice tensor
            logger.debug(f"Initial voice tensor shape: {voice_tensor.shape}")
            
            # Handle different voice tensor formats
            logger.debug(f"Initial voice tensor shape: {voice_tensor.shape}")
            
            # For v0.19 format: [510, 1, 256]
            if voice_tensor.dim() == 3 and voice_tensor.size(1) == 1:
                # Select embedding based on text length
                voice_tensor = voice_tensor[len(text)-1]  # [510, 1, 256] -> [1, 256]
            
            # For v1.0 format: [510, 256]
            elif voice_tensor.dim() == 2 and voice_tensor.size(-1) == 256:
                # Select embedding based on text length
                voice_tensor = voice_tensor[len(text)-1].unsqueeze(0)  # [510, 256] -> [1, 256]
            
            else:
                raise RuntimeError(f"Unsupported voice tensor shape: {voice_tensor.shape}")
            
            logger.debug(f"After reshape voice tensor shape: {voice_tensor.shape}")
            
            logger.debug(f"After reshape voice tensor shape: {voice_tensor.shape}")
            
            # Generate audio directly using KModel
            audio = self.model.forward(
                phonemes=text,  # text is already phonemes from token conversion
                ref_s=voice_tensor,
                speed=speed
            )
            logger.debug(f"Generated audio tensor shape: {audio.shape}")
            yield audio
                
        except Exception as e:
            logger.error(f"Error in KokoroV1Wrapper.forward: {str(e)}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}")

    async def generate(self, tokens: List[int], voice_tensor: torch.Tensor, speed: float = 1.0) -> np.ndarray:
        """Generate audio using KModel's forward pass.
        
        This method provides compatibility with the TTS service interface,
        internally using forward() for generation.
        
        Args:
            tokens: Input token sequence to convert to phonemes
            voice_tensor: Voice embedding tensor (ref_s) containing style information
            speed: Speed multiplier for audio generation
            
        Returns:
            Generated audio as numpy array
            
        Raises:
            RuntimeError: If token-to-phoneme conversion or audio generation fails
        """
        try:
            # Convert tokens back to phonemes using vocab
            phonemes = []
            for token in tokens:
                for p, idx in self.vocab.items():
                    if idx == token:
                        phonemes.append(p)
                        break
            text = ''.join(phonemes)
            logger.debug(f"Decoded tokens to text: '{text[:100]}...'")
            
            # Validate and reshape voice tensor
            logger.debug(f"Initial voice tensor shape: {voice_tensor.shape}")
            
            # Handle different voice tensor formats
            logger.debug(f"Initial voice tensor shape: {voice_tensor.shape}")
            
            # For v0.19 format: [510, 1, 256]
            if voice_tensor.dim() == 3 and voice_tensor.size(1) == 1:
                # Select embedding based on text length
                voice_tensor = voice_tensor[len(text)-1]  # [510, 1, 256] -> [1, 256]
            
            # For v1.0 format: [510, 256]
            elif voice_tensor.dim() == 2 and voice_tensor.size(-1) == 256:
                # Select embedding based on text length
                voice_tensor = voice_tensor[len(text)-1].unsqueeze(0)  # [510, 256] -> [1, 256]
            
            else:
                raise RuntimeError(f"Unsupported voice tensor shape: {voice_tensor.shape}")
            
            logger.debug(f"After reshape voice tensor shape: {voice_tensor.shape}")
            
            logger.debug(f"After reshape voice tensor shape: {voice_tensor.shape}")
            
            try:
                # Generate audio directly using KModel
                audio = self.model.forward(
                    phonemes=text,
                    ref_s=voice_tensor,
                    speed=speed
                )
                logger.debug(f"Generated audio tensor shape: {audio.shape}")
            except Exception as e:
                raise RuntimeError(f"Model forward pass failed: {e}. Input shapes: voice={voice_tensor.shape}, text_len={len(text)}")
            
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
                
            return audio
            
        except Exception as e:
            logger.error(f"Error in KokoroV1Wrapper.generate: {str(e)}")
            raise RuntimeError(f"Failed to generate audio: {str(e)}")


async def build_model(config_path: str = None, model_path: str = None) -> KokoroV1Wrapper:
    """Build a v1.0 model instance.
    
    Args:
        config_path: Optional path to config.json. If None, uses default in builds/v1_0/
        model_path: Optional path to model weights. If None, uses default in models/v1_0/
        
    Returns:
        Initialized KokoroV1Wrapper instance
        
    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.json")
        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent / "models/v1_0/kokoro-v1_0.pth")
            
        logger.info(f"Initializing KokoroV1Wrapper with:")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Model: {model_path}")
        
        if not Path(config_path).exists():
            raise RuntimeError(f"Config file not found: {config_path}")
        if not Path(model_path).exists():
            raise RuntimeError(f"Model file not found: {model_path}")
            
        wrapper = KokoroV1Wrapper(
            config_path=config_path,
            model_path=model_path
        )
        logger.info("Successfully initialized KokoroV1Wrapper")
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to initialize KokoroV1Wrapper: {str(e)}")
        raise RuntimeError(f"Failed to initialize v1.0 model: {str(e)}")