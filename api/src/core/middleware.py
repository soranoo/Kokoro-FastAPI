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

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and manage JWT cookie for user identification
        
        This middleware:
        1. Checks if user has a valid JWT cookie
        2. If not, generates a new user ID and creates a JWT cookie
        3. If yes, validates the JWT and extracts user ID
        4. Automatically refreshes the token if it's close to expiring
        5. Stores user ID in request.state for use by other endpoints
        """
        user_id = None
        jwt_secret = settings.get_jwt_secret()
        cookie_name = settings.jwt_cookie_name
        should_refresh = False
        
        # Try to get existing JWT cookie
        jwt_token = request.cookies.get(cookie_name)
        
        if jwt_token:
            try:
                # Decode and validate JWT
                payload = jwt.decode(jwt_token, jwt_secret, algorithms=["HS256"])
                user_id = payload.get("user_id")
                
                # Validate expiry and check if refresh needed
                exp = payload.get("exp")
                iat = payload.get("iat")
                
                if exp:
                    current_time = datetime.utcnow()
                    expiry_time = datetime.utcfromtimestamp(exp)
                    
                    # Check if token is expired
                    if expiry_time < current_time:
                        logger.debug(f"JWT token expired for user {user_id}")
                        user_id = None
                        should_refresh = True
                    else:
                        # Check if token needs refresh based on threshold
                        if iat:
                            issued_time = datetime.utcfromtimestamp(iat)
                            total_lifetime = (expiry_time - issued_time).total_seconds()
                            remaining_lifetime = (expiry_time - current_time).total_seconds()
                            
                            # Calculate percentage of remaining life
                            if total_lifetime > 0:
                                remaining_percentage = remaining_lifetime / total_lifetime
                                
                                # Refresh if below threshold
                                if remaining_percentage < settings.jwt_refresh_threshold:
                                    should_refresh = True
                                    logger.debug(
                                        f"JWT token for user {user_id} will be refreshed "
                                        f"(remaining: {remaining_percentage:.1%}, threshold: {settings.jwt_refresh_threshold:.1%})"
                                    )
                        
                        logger.debug(f"Valid JWT found for user: {user_id}")
                    
            except jwt.InvalidTokenError as e:
                logger.debug(f"Invalid JWT token: {e}")
                user_id = None
                should_refresh = True
            except Exception as e:
                logger.warning(f"Error decoding JWT: {e}")
                user_id = None
                should_refresh = True
        
        # Generate new user ID if needed
        if not user_id:
            user_id = str(uuid.uuid4())
            logger.debug(f"Generated new user ID: {user_id}")
            should_refresh = True
        
        # Store user ID in request state for access by endpoints
        request.state.user_id = user_id
        
        # Process the request
        response = await call_next(request)
        
        # Set or refresh JWT cookie if needed
        if should_refresh or not jwt_token:
            # Create new JWT token
            expiry = datetime.utcnow() + timedelta(seconds=settings.jwt_cookie_max_age)
            token_payload = {
                "user_id": user_id,
                "exp": expiry,
                "iat": datetime.utcnow()
            }
            new_token = jwt.encode(token_payload, jwt_secret, algorithm="HS256")
            
            # Set cookie in response
            response.set_cookie(
                key=cookie_name,
                value=new_token,
                max_age=settings.jwt_cookie_max_age,
                httponly=True,  # Prevent JavaScript access for security
                secure=False,  # Set to True if using HTTPS
                samesite="lax"  # CSRF protection
            )
            logger.debug(f"Set/refreshed JWT cookie for user: {user_id}")
        
        return response

