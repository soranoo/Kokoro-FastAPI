"""Custom middleware for Kokoro FastAPI"""

import uuid
from datetime import datetime, timedelta
from typing import Callable

import jwt
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

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

        # Add path prefix if set
        if settings.api_url_prefix:
            prefix = "/" + settings.api_url_prefix.strip("/")
            public_paths = {prefix + path for path in public_paths}
            if public_prefixes:
                public_prefixes = tuple(prefix + p for p in public_prefixes)

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


class JWTCookieMiddleware(BaseHTTPMiddleware):
    """Middleware to manage JWT cookies for user session tracking"""

    def _should_skip_jwt_processing(self, request: Request) -> bool:
        """Check if JWT processing should be skipped for this request"""
        return request.method == "OPTIONS" or request.url.path == "/health"

    async def _get_or_create_user_id(self, request: Request) -> tuple[str, bool, float | None]:
        """Get existing user ID from JWT or create new one. Returns (user_id, should_refresh, session_expiry)"""
        jwt_secret = settings.get_jwt_secret()
        cookie_name = settings.jwt_cookie_name
        user_id = None
        should_refresh = False
        session_expiry_timestamp = None

        # Try to get existing JWT cookie
        jwt_token = request.cookies.get(cookie_name)

        if jwt_token:
            user_id, should_refresh, session_expiry_timestamp = self._validate_jwt_token(jwt_token, jwt_secret)

        # Generate new user ID if needed
        if not user_id:
            user_id = str(uuid.uuid4())
            logger.debug(f"Generated new user ID: {user_id}")
            should_refresh = True

        return user_id, should_refresh, session_expiry_timestamp

    def _validate_jwt_token(self, jwt_token: str, jwt_secret: str) -> tuple[str | None, bool, float | None]:
        """Validate JWT token and return (user_id, should_refresh, session_expiry)"""
        try:
            # Decode and validate JWT
            payload = jwt.decode(jwt_token, jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")

            # Validate expiry and check if refresh needed
            exp = payload.get("exp")
            iat = payload.get("iat")
            should_refresh = False
            session_expiry_timestamp = None

            if exp:
                current_time = datetime.utcnow()
                expiry_time = datetime.utcfromtimestamp(exp)
                session_expiry_timestamp = exp

                # Check if token is expired
                if expiry_time < current_time:
                    logger.debug(f"JWT token expired for user {user_id}")
                    return None, True, None
                else:
                    # Check if token needs refresh based on threshold
                    should_refresh = self._should_refresh_token(iat, expiry_time, current_time, user_id)

                    logger.debug(f"Valid JWT found for user: {user_id}")
                    return user_id, should_refresh, session_expiry_timestamp

        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid JWT token: {e}")
        except Exception as e:
            logger.warning(f"Error decoding JWT: {e}")

        return None, True, None

    def _should_refresh_token(self, iat: float, expiry_time: datetime, current_time: datetime, user_id: str) -> bool:
        """Check if JWT token should be refreshed based on remaining lifetime"""
        if not iat:
            return False

        issued_time = datetime.utcfromtimestamp(iat)
        total_lifetime = (expiry_time - issued_time).total_seconds()
        remaining_lifetime = (expiry_time - current_time).total_seconds()

        if total_lifetime <= 0:
            return False

        remaining_percentage = remaining_lifetime / total_lifetime

        # Refresh if below threshold
        if remaining_percentage < settings.jwt_refresh_threshold:
            logger.debug(
                f"JWT token for user {user_id} will be refreshed "
                f"(remaining: {remaining_percentage:.1%}, threshold: {settings.jwt_refresh_threshold:.1%})"
            )
            return True

        return False

    def _create_jwt_token(self, user_id: str, jwt_secret: str) -> tuple[str, float]:
        """Create new JWT token and return (token, expiry_timestamp)"""
        expiry = datetime.utcnow() + timedelta(seconds=settings.jwt_cookie_max_age)
        session_expiry_timestamp = expiry.timestamp()
        token_payload = {
            "user_id": user_id,
            "exp": int(expiry.timestamp()),
            "iat": int(datetime.utcnow().timestamp())
        }
        new_token = jwt.encode(token_payload, jwt_secret, algorithm="HS256")
        return new_token, session_expiry_timestamp

    def _set_jwt_cookie(self, response: Response, token: str, user_id: str):
        """Set JWT cookie in response"""
        response.set_cookie(
            key=settings.jwt_cookie_name,
            value=token,
            max_age=settings.jwt_cookie_max_age,
            httponly=True,  # Prevent JavaScript access for security
            secure=settings.jwt_cookie_secure,
            samesite="lax"  # CSRF protection
        )
        logger.debug(f"Set/refreshed JWT cookie for user: {user_id}")

    async def _track_session_in_redis(self, request: Request, user_id: str, session_expiry_timestamp: float):
        """Track user session in Redis for cleanup"""
        if not hasattr(request.app.state, 'redis') or not request.app.state.redis:
            return

        try:
            from ..services.temp_manager import track_user_session
            await track_user_session(request.app.state.redis, user_id, session_expiry_timestamp)
        except Exception as e:
            logger.warning(f"Failed to track session in Redis: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and manage JWT cookie for user identification"""

        if self._should_skip_jwt_processing(request):
            return await call_next(request)

        # Get or create user ID
        user_id, should_refresh, session_expiry_timestamp = await self._get_or_create_user_id(request)

        # Store user ID in request state for access by endpoints
        request.state.user_id = user_id

        # Process the request
        response = await call_next(request)

        # Set or refresh JWT cookie if needed
        if should_refresh:
            new_token, session_expiry_timestamp = self._create_jwt_token(user_id, settings.get_jwt_secret())
            self._set_jwt_cookie(response, new_token, user_id)

        # Track session expiry in Redis for session-based cleanup
        if session_expiry_timestamp:
            await self._track_session_in_redis(request, user_id, session_expiry_timestamp)

        return response


