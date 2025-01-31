"""OpenAI-compatible router with version support"""

import json
import os
from typing import AsyncGenerator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from loguru import logger
from pydantic import BaseModel

from ..services.audio import AudioService
from ..services.tts_service_v2 import TTSService
from ..structures.schemas import OpenAISpeechRequest
from ..core.config import settings


class OpenAISpeechRequestV2(OpenAISpeechRequest):
    """Extended OpenAI speech request with version support."""
    version: Optional[str] = None  # "v0.19" or "v1.0"


def load_openai_mappings() -> Dict:
    """Load OpenAI voice and model mappings from JSON"""
    api_dir = os.path.dirname(os.path.dirname(__file__))
    mapping_path = os.path.join(api_dir, "core", "openai_mappings.json")
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}


# Global mappings
_openai_mappings = load_openai_mappings()


router = APIRouter(
    tags=["OpenAI Compatible TTS V2"],
    responses={404: {"description": "Not found"}},
)

# Global TTSService instance with lock
_tts_service = None
_init_lock = None


async def get_tts_service() -> TTSService:
    """Get global TTSService instance"""
    global _tts_service, _init_lock
    
    if _init_lock is None:
        import asyncio
        _init_lock = asyncio.Lock()
    
    if _tts_service is None:
        async with _init_lock:
            if _tts_service is None:
                _tts_service = await TTSService.create()
                logger.info("Created global TTSService V2 instance")
    
    return _tts_service


async def process_voices(
    voice_input: Union[str, List[str]],
    tts_service: TTSService,
    version: Optional[str] = None
) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    if isinstance(voice_input, str):
        mapped_voice = _openai_mappings["voices"].get(voice_input)
        if mapped_voice:
            voice_input = mapped_voice
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        voices = [_openai_mappings["voices"].get(v, v) for v in voice_input]
        voices = [v.strip() for v in voices if v.strip()]

    if not voices:
        raise ValueError("No voices provided")

    if len(voices) == 1:
        available_voices = await tts_service.list_voices(version)
        if voices[0] not in available_voices:
            raise ValueError(
                f"Voice '{voices[0]}' not found for version {version or 'any'}. "
                f"Available voices: {', '.join(sorted(available_voices))}"
            )
        return voices[0]

    # For voice combinations, validate all voices exist for the specified version
    available_voices = await tts_service.list_voices(version)
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Base voice '{voice}' not found for version {version or 'any'}. "
                f"Available voices: {', '.join(sorted(available_voices))}"
            )

    return await tts_service.combine_voices(voices=voices, version=version)


async def stream_audio_chunks(
    tts_service: TTSService, 
    request: OpenAISpeechRequestV2,
    client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    voice_to_use = await process_voices(request.voice, tts_service, request.version)
    
    try:
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_to_use,
            speed=request.speed,
            version=request.version,
            output_format=request.response_format,
        ):
            is_disconnected = client_request.is_disconnected
            if callable(is_disconnected):
                is_disconnected = await is_disconnected()
            if is_disconnected:
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        raise


@router.post("/v2/audio/speech")
async def create_speech(
    request: OpenAISpeechRequestV2,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech with version support"""
    if request.model not in _openai_mappings["models"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}",
                "type": "invalid_request_error"
            }
        )
    
    try:
        tts_service = await get_tts_service()
        voice_to_use = await process_voices(request.voice, tts_service, request.version)

        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        if request.stream:
            generator = stream_audio_chunks(tts_service, request, client_request)
            
            if request.return_download_link:
                from ..services.temp_manager import TempFileWriter
                
                temp_writer = TempFileWriter(request.response_format)
                await temp_writer.__aenter__()
                
                download_path = temp_writer.download_path
                
                headers = {
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Download-Path": download_path
                }

                async def dual_output():
                    try:
                        async for chunk in generator:
                            if chunk:
                                await temp_writer.write(chunk)
                                yield chunk

                        await temp_writer.finalize()
                    except Exception as e:
                        logger.error(f"Error in dual output streaming: {e}")
                        await temp_writer.__aexit__(type(e), e, e.__traceback__)
                        raise
                    finally:
                        if not temp_writer._finalized:
                            await temp_writer.__aexit__(None, None, None)

                return StreamingResponse(
                    dual_output(),
                    media_type=content_type,
                    headers=headers
                )
            
            return StreamingResponse(
                generator,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            audio, _ = await tts_service.generate_audio(
                text=request.input,
                voice=voice_to_use,
                speed=request.speed,
                version=request.version
            )

            content = await AudioService.convert_audio(
                audio, 24000, request.response_format,
                is_first_chunk=True,
                is_last_chunk=True
            )

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error"
            }
        )


@router.get("/v2/audio/versions")
async def list_versions():
    """List available model versions"""
    try:
        tts_service = await get_tts_service()
        return {
            "versions": tts_service.available_versions,
            "current": tts_service.current_version
        }
    except Exception as e:
        logger.error(f"Error listing versions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve version list",
                "type": "server_error"
            }
        )


@router.post("/v2/audio/version")
async def set_version(version: str):
    """Set default model version"""
    try:
        tts_service = await get_tts_service()
        tts_service.set_version(version)
        return {
            "message": f"Set default version to {version}",
            "versions": tts_service.available_versions,
            "current": tts_service.current_version
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except Exception as e:
        logger.error(f"Error setting version: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to set version",
                "type": "server_error"
            }
        )


@router.get("/v2/audio/voices")
async def list_voices(version: Optional[str] = None):
    """List all available voices for text-to-speech.
    
    Args:
        version: Optional version to filter voices by ("v0.19" or "v1.0")
    """
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices(version)
        return {
            "voices": voices,
            "version": version or tts_service.current_version
        }
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error"
            }
        )


@router.post("/v2/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]], version: Optional[str] = None):
    """Combine multiple voices into a new voice.
    
    Args:
        request: Either a string with voices separated by + or a list of voice names
        version: Optional version to filter voices by ("v0.19" or "v1.0")
    """
    try:
        tts_service = await get_tts_service()
        combined_voice = await process_voices(request, tts_service, version)
        voices = await tts_service.list_voices(version)
        return {
            "voices": voices,
            "voice": combined_voice,
            "version": version or tts_service.current_version
        }

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process voice combination request",
                "type": "server_error"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error"
            }
        )