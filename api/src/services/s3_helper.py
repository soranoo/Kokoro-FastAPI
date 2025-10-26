"""S3 helper utilities for temp file storage and signed URL generation"""

import hashlib
import hmac
import time
from typing import Optional, Dict
from loguru import logger

from ..core.config import settings


def generate_s3_key_signature(s3_key: str, secret: str) -> str:
    """Generate HMAC SHA-256 signature for S3 key
    
    Args:
        s3_key: S3 object key
        secret: Secret key for HMAC signing
        
    Returns:
        Hex-encoded HMAC signature
    """
    signature = hmac.new(
        secret.encode('utf-8'),
        s3_key.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


def create_s3_key_with_signature(s3_key: str) -> Dict[str, str]:
    """Create S3 key with HMAC signature for verification
    
    Args:
        s3_key: S3 object key
        
    Returns:
        Dictionary with 'key' and 'signature' fields
    """
    secret = settings.get_s3_signature_secret()
    if not secret:
        logger.error("S3 signature secret not configured")
        return {"key": s3_key, "signature": ""}
    
    signature = generate_s3_key_signature(s3_key, secret)
    return {
        "key": s3_key,
        "signature": signature
    }


def verify_s3_key_signature(s3_key: str, signature: str) -> bool:
    """Verify S3 key signature
    
    Args:
        s3_key: S3 object key
        signature: HMAC signature to verify
        
    Returns:
        True if signature is valid, False otherwise
    """
    secret = settings.get_s3_signature_secret()
    if not secret:
        logger.error("S3 signature secret not configured")
        return False
    
    expected_signature = generate_s3_key_signature(s3_key, secret)
    return hmac.compare_digest(signature, expected_signature)


def generate_s3_presigned_url(s3_client, s3_key: str, expiry_seconds: Optional[int] = None) -> Optional[str]:
    """Generate presigned URL for S3 object download
    
    Args:
        s3_client: Boto3 S3 client
        s3_key: S3 object key
        expiry_seconds: URL expiry time in seconds (default: from settings)
        
    Returns:
        Presigned URL string or None if error
    """
    if not s3_client:
        logger.error("S3 client not configured")
        return None
    
    if expiry_seconds is None:
        expiry_seconds = settings.s3_signed_url_expiry
    
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.s3_bucket_name,
                'Key': s3_key
            },
            ExpiresIn=expiry_seconds
        )
        return presigned_url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
        return None


async def upload_to_s3(s3_client, s3_key: str, data: bytes) -> bool:
    """Upload data to S3
    
    Args:
        s3_client: Boto3 S3 client
        s3_key: S3 object key
        data: Binary data to upload
        
    Returns:
        True if upload successful, False otherwise
    """
    if not s3_client:
        logger.error("S3 client not configured")
        return False
    
    try:
        s3_client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
            Body=data
        )
        logger.debug(f"Uploaded {len(data)} bytes to S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3 {s3_key}: {e}")
        return False


async def delete_from_s3(s3_client, s3_key: str) -> bool:
    """Delete object from S3
    
    Args:
        s3_client: Boto3 S3 client
        s3_key: S3 object key
        
    Returns:
        True if deletion successful, False otherwise
    """
    if not s3_client:
        logger.error("S3 client not configured")
        return False
    
    try:
        s3_client.delete_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key
        )
        logger.debug(f"Deleted from S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete from S3 {s3_key}: {e}")
        return False


async def check_s3_object_exists(s3_client, s3_key: str) -> bool:
    """Check if S3 object exists
    
    Args:
        s3_client: Boto3 S3 client
        s3_key: S3 object key
        
    Returns:
        True if object exists, False otherwise
    """
    if not s3_client:
        logger.error("S3 client not configured")
        return False
    
    try:
        s3_client.head_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key
        )
        return True
    except Exception as e:
        # Object doesn't exist or other error
        logger.debug(f"S3 object not found: {s3_key}")
        return False
