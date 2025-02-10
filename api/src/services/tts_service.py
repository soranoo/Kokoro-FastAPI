"""TTS service using model and voice managers."""

import asyncio
import os
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.kokoro_v1 import KokoroV1
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split


class TTSService:
    """Text-to-speech service."""

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None
        # Create request queue for global request management
        self._request_queue = asyncio.Queue(maxsize=32)

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        chunk_text: str,
        tokens: List[int],
        voice_name: str,
        voice_path: str,
        speed: float,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[Union[np.ndarray, bytes], None]:
        """Process tokens into audio."""
        try:
            # Handle stream finalization
            if is_last:
                # Skip format conversion for raw audio mode
                if not output_format:
                    yield np.array([], dtype=np.float32)
                    return

                result = await AudioService.convert_audio(
                    np.array([0], dtype=np.float32),  # Dummy data for type checking
                    24000,
                    output_format,
                    is_first_chunk=False,
                    normalizer=normalizer,
                    is_last_chunk=True,
                )
                yield result
                return

            # Skip empty chunks
            if not tokens and not chunk_text:
                return

            # Generate audio using model pool
            async for chunk_audio in self.model_manager.generate(
                chunk_text,
                (voice_name, voice_path),
                speed=speed,
                lang_code=lang_code,
            ):
                # For streaming, convert to bytes
                if output_format:
                    try:
                        converted = await AudioService.convert_audio(
                            chunk_audio,
                            24000,
                            output_format,
                            is_first_chunk=is_first,
                            normalizer=normalizer,
                            is_last_chunk=is_last,
                        )
                        yield converted
                    except Exception as e:
                        logger.error(f"Failed to convert audio: {str(e)}")
                else:
                    yield chunk_audio

        except Exception as e:
            logger.error(f"Failed to process tokens: {str(e)}")

    async def _get_voice_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Check if it's a combined voice
            if "+" in voice:
                # Split on + but preserve any parentheses
                voice_parts = []
                weights = []
                for part in voice.split("+"):
                    part = part.strip()
                    if not part:
                        continue
                    # Extract voice name and weight if present
                    if "(" in part and ")" in part:
                        voice_name = part.split("(")[0].strip()
                        weight = float(part.split("(")[1].split(")")[0])
                    else:
                        voice_name = part
                        weight = 1.0
                    voice_parts.append(voice_name)
                    weights.append(weight)

                if len(voice_parts) < 2:
                    raise RuntimeError(f"Invalid combined voice name: {voice}")

                # Normalize weights to sum to 1
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Load and combine voices
                voice_tensors = []
                for v, w in zip(voice_parts, weights):
                    path = await self._voice_manager.get_voice_path(v)
                    if not path:
                        raise RuntimeError(f"Voice not found: {v}")
                    logger.debug(f"Loading voice tensor from: {path}")
                    voice_tensor = torch.load(path, map_location="cpu")
                    voice_tensors.append(voice_tensor * w)

                # Sum the weighted voice tensors
                logger.debug(
                    f"Combining {len(voice_tensors)} voice tensors with weights {weights}"
                )
                combined = torch.sum(torch.stack(voice_tensors), dim=0)

                # Save combined tensor
                temp_dir = tempfile.gettempdir()
                combined_path = os.path.join(temp_dir, f"{voice}.pt")
                logger.debug(f"Saving combined voice to: {combined_path}")
                torch.save(combined, combined_path)

                return voice, combined_path
            else:
                # Single voice
                path = await self._voice_manager.get_voice_path(voice)
                if not path:
                    raise RuntimeError(f"Voice not found: {voice}")
                logger.debug(f"Using single voice path: {path}")
                return voice, path
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate and stream audio chunks."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0

        try:
            # Get voice path, handling combined voices
            voice_name, voice_path = await self._get_voice_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
            )

            # Process text in chunks with smart splitting
            async for chunk_text, tokens in smart_split(text):
                try:
                    # Process audio for chunk
                    async for result in self._process_chunk(
                        chunk_text,  # Pass text for Kokoro V1
                        tokens,  # Pass tokens for legacy backends
                        voice_name,  # Pass voice name
                        voice_path,  # Pass voice path
                        speed,
                        output_format,
                        is_first=(chunk_index == 0),
                        is_last=False,  # We'll update the last chunk later
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                    ):
                        if result is not None:
                            yield result
                            chunk_index += 1
                        else:
                            logger.warning(
                                f"No audio generated for chunk: '{chunk_text[:100]}...'"
                            )

                except Exception as e:
                    logger.error(
                        f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}"
                    )
                    continue

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    # Empty tokens list to finalize audio
                    async for result in self._process_chunk(
                        "",  # Empty text
                        [],  # Empty tokens
                        voice_name,
                        voice_path,
                        speed,
                        output_format,
                        is_first=False,
                        is_last=True,  # Signal this is the last chunk
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                    ):
                        if result is not None:
                            yield result
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise

    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        return_timestamps: bool = False,
        lang_code: Optional[str] = None,
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, List[dict]]]:
        """Generate complete audio for text using streaming internally."""
        start_time = time.time()
        chunks = []
        word_timestamps = []

        try:
            # Get voice path
            voice_name, voice_path = await self._get_voice_path(voice)

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in text chunking"
            )

            # Process text in chunks
            async for chunk_text, tokens in smart_split(text):
                # Generate audio for chunk using model pool
                async for chunk_audio in self.model_manager.generate(
                    chunk_text,
                    (voice_name, voice_path),
                    speed=speed,
                    lang_code=pipeline_lang_code,
                ):
                    chunks.append(chunk_audio)

            if not chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Combine chunks
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            processing_time = time.time() - start_time

            # Return with timestamps if requested
            if return_timestamps:
                return audio, processing_time, word_timestamps
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices.

        Returns:
            Combined voice tensor
        """
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()

    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate audio directly from phonemes.

        Args:
            phonemes: Phonemes in Kokoro format
            voice: Voice name
            speed: Speed multiplier
            lang_code: Optional language code override

        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        try:
            # Get voice path
            voice_name, voice_path = await self._get_voice_path(voice)

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline"
            )

            # Generate audio using model pool
            chunks = []
            async for chunk_audio in self.model_manager.generate(
                phonemes,
                (voice_name, voice_path),
                speed=speed,
                lang_code=pipeline_lang_code,
            ):
                chunks.append(chunk_audio)

            if not chunks:
                raise ValueError("No audio generated")

            # Combine chunks
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise
