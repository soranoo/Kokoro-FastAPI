"""TTS service with version support."""

import time
from typing import List, Tuple, Optional, AsyncGenerator, Union

import numpy as np
import torch
from loguru import logger

from ..core.config import settings
from ..inference.version_manager import get_version_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing.text_processor import process_text_chunk, smart_split
from .text_processing import tokenize


class TTSService:
    """Text-to-speech service with version support."""

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.version_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> 'TTSService':
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.version_manager = await get_version_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        tokens: List[int],
        voice_tensor: torch.Tensor,
        speed: float,
        version: Optional[str] = None,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
    ) -> AsyncGenerator[Union[np.ndarray, bytes], None]:
        """Process tokens into audio."""
        try:
            # Handle stream finalization
            if is_last:
                if not output_format:
                    yield np.array([], dtype=np.float32)
                    return
                
                final_chunk = await AudioService.convert_audio(
                    np.array([0], dtype=np.float32),
                    24000,
                    output_format,
                    is_first_chunk=False,
                    normalizer=normalizer,
                    is_last_chunk=True
                )
                if final_chunk is not None:
                    yield final_chunk
                return
            
            # Skip empty chunks
            if not tokens:
                return

            # Get model for specified version
            model = await self.version_manager.get_model(version)

            if version == "v1.0":
                # For v1.0, we need to handle the generator
                try:
                    # Split long sequences to avoid index out of bounds
                    max_length = 500  # v1.0 model context limit
                    if len(tokens) > max_length:
                        logger.warning(f"Truncating sequence from {len(tokens)} to {max_length} tokens")
                        tokens = tokens[:max_length]
                    
                    # Process all chunks from the generator
                    async for audio in model.forward(tokens, voice_tensor, speed=speed):
                        if audio is None:
                            continue
                        
                        # Convert tensor to numpy if needed
                        if isinstance(audio, torch.Tensor):
                            audio = audio.cpu().numpy()
                        
                        # Convert audio if needed
                        if output_format:
                            converted = await AudioService.convert_audio(
                                audio,
                                24000,
                                output_format,
                                is_first_chunk=is_first,
                                normalizer=normalizer,
                                is_last_chunk=is_last
                            )
                            if converted is not None:
                                yield converted
                        else:
                            yield audio
                except Exception as e:
                    logger.error(f"Generation failed: {str(e)}")
                    return
            else:
                # For v0.19, use existing generate method
                audio = await model.generate(tokens, voice_tensor, speed=speed)
                
                if audio is None:
                    logger.error("Model generated None for audio chunk")
                    return
                
                if len(audio) == 0:
                    logger.error("Model generated empty audio chunk")
                    return
                    
                # Convert audio if needed
                if output_format:
                    try:
                        converted = await AudioService.convert_audio(
                            audio,
                            24000,
                            output_format,
                            is_first_chunk=is_first,
                            normalizer=normalizer,
                            is_last_chunk=is_last
                        )
                        if converted is not None:
                            yield converted
                    except Exception as e:
                        logger.error(f"Failed to convert audio: {str(e)}")
                        return
                else:
                    yield audio

        except Exception as e:
            logger.error(f"Failed to process tokens: {str(e)}")
            return

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        version: Optional[str] = None,
        output_format: str = "wav",
    ) -> AsyncGenerator[bytes, None]:
        """Generate and stream audio chunks."""
        stream_normalizer = AudioNormalizer()
        voice_tensor = None
        chunk_index = 0
        
        try:
            # Get model and load voice
            model = await self.version_manager.get_model(version)
            device = "cuda" if settings.use_gpu else "cpu"
            voice_tensor = await self._voice_manager.load_voice(voice, device=device, version=version)

            # Process text in chunks with smart splitting
            async for chunk_text, tokens in smart_split(text):
                try:
                    # Process audio for chunk
                    async for result in self._process_chunk(
                        tokens,
                        voice_tensor,
                        speed,
                        version=version,
                        output_format=output_format,
                        is_first=(chunk_index == 0),
                        is_last=False,
                        normalizer=stream_normalizer
                    ):
                        if result is not None:
                            yield result
                            chunk_index += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}")
                    continue

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    async for final_result in self._process_chunk(
                        [],
                        voice_tensor,
                        speed,
                        version=version,
                        output_format=output_format,
                        is_first=False,
                        is_last=True,
                        normalizer=stream_normalizer
                    ):
                        if final_result is not None:
                            logger.debug("Yielding final chunk to finalize audio")
                            yield final_result
                except Exception as e:
                    logger.error(f"Failed to process final chunk: {str(e)}")
            else:
                logger.warning("No audio chunks were successfully processed")

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise
        finally:
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        version: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate complete audio for text using streaming internally."""
        start_time = time.time()
        chunks = []
        voice_tensor = None
        
        try:
            # Get model and load voice
            model = await self.version_manager.get_model(version)
            device = "cuda" if settings.use_gpu else "cpu"
            voice_tensor = await self._voice_manager.load_voice(voice, device=device, version=version)

            if version == "v1.0":
                # For v1.0, use streaming internally
                async for chunk in self.generate_audio_stream(
                    text, voice, speed, version, output_format=None
                ):
                    if chunk is not None and isinstance(chunk, np.ndarray):
                        chunks.append(chunk)
            else:
                # For v0.19, use direct generation
                async for chunk_text, tokens in smart_split(text):
                    try:
                        audio = await model.generate(tokens, voice_tensor, speed=speed)
                        if audio is not None:
                            chunks.append(audio)
                    except Exception as e:
                        logger.error(f"Failed to generate audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}")
                        continue

            if not chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Concatenate chunks
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise
        finally:
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def combine_voices(self, voices: List[str], version: Optional[str] = None) -> str:
        """Combine multiple voices.
        
        Args:
            voices: List of voice names to combine
            version: Optional version to filter voices by
        """
        return await self._voice_manager.combine_voices(voices, version=version)

    async def list_voices(self, version: Optional[str] = None) -> List[str]:
        """List available voices.
        
        Args:
            version: Optional version to filter voices by
        """
        return await self._voice_manager.list_voices(version)

    def set_version(self, version: str) -> None:
        """Set default model version."""
        self.version_manager.set_default_version(version)

    @property
    def current_version(self) -> str:
        """Get current model version."""
        return self.version_manager.current_version

    @property
    def available_versions(self) -> List[str]:
        """Get list of available model versions."""
        return self.version_manager.available_versions