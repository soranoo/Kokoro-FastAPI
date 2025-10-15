"""
FastAPI OpenAI Compatible API
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .core.middleware import (BearerAuthMiddleware)
from .routers.debug import router as debug_router
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .routers.web_player import router as web_router


from .structures.schemas import HealthCheckResponse, TestEndpointResponse


def setup_logger():
    """Configure loguru logger with custom formatting"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "<fg #4169E1>{module}:{line}</fg #4169E1> | "
                "{message}",
                "colorize": True,
                "level": "DEBUG",
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization and Redis cleanup"""
    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager
    from .services.temp_manager import cleanup_temp_files, redis_periodic_cleanup_loop

    # Initialize Redis client and cleanup task if configured
    redis_client = None
    redis_cleanup_task = None
    
    if settings.redis_host and settings.redis_port:
        try:
            redis_client = settings.get_redis_client()
            await redis_client.ping()
            logger.info(f"✅ Redis connected: {settings.redis_host}:{settings.redis_port}")
            
            # Start background cleanup task
            redis_cleanup_task = asyncio.create_task(redis_periodic_cleanup_loop(redis_client))
            logger.info("🔄 Redis temp file cleanup task started")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, falling back to filesystem cleanup: {e}")
            redis_client = None
            if redis_cleanup_task:
                redis_cleanup_task.cancel()
                redis_cleanup_task = None

    # Clean old temp files on startup (filesystem cleanup if Redis not available)
    if settings.enable_temp_file_system or not redis_client:
        await cleanup_temp_files()
    
    # Store Redis client in app state for use in endpoints
    app.state.redis = redis_client

    logger.info("Loading TTS model and voice packs...")

    try:
        # Initialize managers
        model_manager = await get_manager()
        voice_manager = await get_voice_manager()

        # Initialize model with warmup and get status
        device, model, voicepack_count = await model_manager.initialize_with_warmup(
            voice_manager
        )

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    boundary = "░" * 2 * 12
    startup_msg = f"""

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │ 
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                """
    startup_msg += f"\nModel warmed up on {device}: {model}"
    if device == "mps":
        startup_msg += "\nUsing Apple Metal Performance Shaders (MPS)"
    elif device == "cuda":
        startup_msg += f"\nCUDA: {torch.cuda.is_available()}"
    else:
        startup_msg += "\nRunning on CPU"
    startup_msg += f"\n{voicepack_count} voice packs loaded"

    # Add security info
    if settings.api_bearer_token:
        startup_msg += "\n\n🔒 Authentication: ENABLED (Bearer token required)"
    else:
        startup_msg += "\n\n🔓 Authentication: DISABLED (Open access)"
    
    if settings.hide_server_header:
        startup_msg += "\n🛡️  Server header: HIDDEN"

    # Add OpenAPI docs info
    if settings.enable_openapi_docs:
        startup_msg += f"\n📚 API Docs: http://localhost:{settings.port}/docs"
    else:
        startup_msg += "\n📚 API Docs: DISABLED"

    # Add web player info if enabled
    if settings.enable_web_player:
        startup_msg += (
            f"\n🎵 Web Player: http://{settings.host}:{settings.port}/web/"
        )
        startup_msg += f"\n   or http://localhost:{settings.port}/web/"
    else:
        startup_msg += "\n🎵 Web Player: DISABLED"
    
    # Add temp file management info
    if redis_client:
        startup_msg += f"\n🗂️  Temp Files: Redis-managed (TTL: {settings.temp_file_ttl_seconds}s)"
    elif settings.enable_temp_file_system:
        startup_msg += "\n🗂️  Temp Files: Filesystem cleanup enabled"
    else:
        startup_msg += "\n🗂️  Temp Files: No cleanup configured"

    startup_msg += f"\n{boundary}\n"
    logger.info(startup_msg)

    yield

    # Shutdown: cleanup Redis resources
    if redis_cleanup_task:
        logger.info("Stopping Redis cleanup task...")
        redis_cleanup_task.cancel()
        try:
            await redis_cleanup_task
        except asyncio.CancelledError:
            pass
    
    if redis_client:
        logger.info("Closing Redis connection...")
        await redis_client.close()

# Include routers
# Normalize API URL prefix: ensure leading slash and no trailing slash when provided
raw_prefix = (settings.api_url_prefix or "").strip()
if raw_prefix:
    api_prefix = "/" + raw_prefix.strip("/")
else:
    api_prefix = ""

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    # Conditionally enable/disable OpenAPI documentation
    openapi_url=f"{api_prefix}/openapi.json" if settings.enable_openapi_docs else None,
    docs_url=f"{api_prefix}/docs" if settings.enable_openapi_docs else None,
    redoc_url=f"{api_prefix}/redoc" if settings.enable_openapi_docs else None,
)

# 1. Authentication middleware
app.add_middleware(BearerAuthMiddleware)

# 2. CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(openai_router, prefix=f"{api_prefix}/v1")

app.include_router(dev_router, prefix=api_prefix)  # Development endpoints
app.include_router(debug_router, prefix=api_prefix)  # Debug endpoints

if settings.enable_web_player:
    app.include_router(web_router, prefix=f"{api_prefix}/web")  # Web player static files


# Health check endpoint
@app.get(f"{api_prefix}/health", tags=["Health"], response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint
    
    Returns the health status of the API service.
    
    Returns:
        Dictionary with status field indicating service health
    """
    return HealthCheckResponse(status="healthy")


@app.get(f"{api_prefix}/test", tags=["Testing"], response_model=TestEndpointResponse)
async def test_endpoint() -> TestEndpointResponse:
    """Test endpoint to verify routing
    
    Simple test endpoint to verify that the API routing is working correctly.
    
    Returns:
        Dictionary with status field
    """
    return TestEndpointResponse(status="ok")


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True, server_header=not settings.hide_server_header)
