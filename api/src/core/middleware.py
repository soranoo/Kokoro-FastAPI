"""Custom middleware for Kokoro FastAPI"""

from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

class DisabledEndpointsMiddleware(BaseHTTPMiddleware):
    """Middleware to block access to disabled features"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Block OpenAPI docs if disabled
        if not settings.enable_openapi_docs:
            if path in {"/docs", "/redoc", "/openapi.json"}:
                logger.warning(f"Access denied to disabled endpoint: {path}")
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "error": "not_found",
                        "message": "API documentation is disabled",
                        "type": "feature_disabled",
                    },
                )

        # Block web player if disabled
        if not settings.enable_web_player:
            if path.startswith("/web/"):
                logger.warning(f"Access denied to disabled web player: {path}")
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "error": "not_found",
                        "message": "Web player is disabled",
                        "type": "feature_disabled",
                    },
                )

        return await call_next(request)

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce Bearer token authentication"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication if no token is configured
        if not settings.api_bearer_token:
            return await call_next(request)

        # Build dynamic public paths based on settings
        public_paths = {"/health"}
        
        # Add OpenAPI paths if enabled
        if settings.enable_openapi_docs:
            public_paths.update({"/docs", "/redoc", "/openapi.json"})
        
        # Add web player prefix if enabled
        public_prefixes = ()
        if settings.enable_web_player:
            public_prefixes = ("/web/",)

        # Allow public paths
        if request.url.path in public_paths:
            return await call_next(request)

        # Allow public prefixes
        if public_prefixes and request.url.path.startswith(public_prefixes):
            return await call_next(request)

        # Check for Authorization header
        authorization = request.headers.get("authorization")

        if not authorization:
            logger.warning(f"Missing authorization header for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Missing authorization header",
                    "type": "authentication_error",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse Bearer token
        try:
            scheme, token = authorization.split(None, 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
        except ValueError:
            logger.warning(
                f"Invalid authorization header format for {request.url.path}: {authorization}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Invalid authorization header format. Expected 'Bearer <token>'",
                    "type": "authentication_error",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify token
        if token != settings.api_bearer_token:
            logger.warning(f"Invalid bearer token for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Invalid bearer token",
                    "type": "authentication_error",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Token is valid, proceed with request
        return await call_next(request)
