# Session-Based File Cleanup

## Overview

The Kokoro FastAPI application implements **session-based cleanup** to automatically delete all temporary files when a user's JWT session expires. This ensures that:

1. **Security**: Users cannot access files after their session expires
2. **Storage Management**: Orphaned files are automatically cleaned up
3. **Privacy**: User data is removed when the session ends

## How It Works

### 1. Session Tracking

When a user makes a request:

1. **JWT Middleware** (`JWTCookieMiddleware`) validates or creates a JWT cookie
2. The middleware extracts the session expiry timestamp from the JWT
3. Session expiry is tracked in Redis:
   - Key: `temp_files:sessions` (sorted set)
   - Value: `user_id` → `expiry_timestamp`

### 2. File Registration

When a temporary file is created:

1. File path is registered in Redis sorted set: `temp_files:files`
2. User ownership is tracked: `temp_files:files:ownership`
3. Storage type is tracked: `temp_files:files:storage_type` (local or S3)
4. **User-file mapping** is created: `temp_files:files:user_files:{user_id}` (set of file paths)

### 3. Session-Based Cleanup

The `redis_periodic_cleanup_loop()` runs two cleanup operations:

#### A. Time-Based Cleanup (Existing)
- Removes files that have exceeded their TTL
- Checks `temp_files:files` sorted set for expired timestamps

#### B. Session-Based Cleanup (New)
- Checks `temp_files:sessions` for expired sessions
- For each expired session:
  1. Gets all files owned by the user from `temp_files:files:user_files:{user_id}`
  2. Deletes each file from storage (local filesystem or S3)
  3. Removes file from Redis tracking structures
  4. Cleans up session record

### 4. Cleanup Flow

```
┌─────────────────────────────────────────────────────────────┐
│  redis_periodic_cleanup_loop()                              │
│  (runs every TEMP_REDIS_CLEANUP_INTERVAL_SECONDS)           │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─► redis_cleanup_once()
             │   └─► Deletes files past their TTL
             │
             └─► cleanup_expired_sessions()
                 ├─► Find expired sessions in Redis
                 ├─► For each expired user_id:
                 │   ├─► Get user's files from user_files:{user_id}
                 │   ├─► Delete from S3 or local filesystem
                 │   └─► Remove from Redis tracking
                 └─► Clean up session records
```

## Redis Data Structures

### Session Tracking
```
temp_files:sessions (sorted set)
{
  "user-uuid-1": 1704067200.0,  # Unix timestamp
  "user-uuid-2": 1704070800.0
}
```

### User Files Mapping
```
temp_files:files:user_files:{user_id} (set)
{
  "/tmp/audio_abc123.wav",
  "/tmp/audio_def456.wav",
  "s3/audio/xyz789.wav"
}
```

### File Ownership
```
temp_files:files:ownership (hash)
{
  "/tmp/audio_abc123.wav": "user-uuid-1",
  "s3/audio/xyz789.wav": "user-uuid-2"
}
```

### Storage Type
```
temp_files:files:storage_type (hash)
{
  "/tmp/audio_abc123.wav": "local",
  "s3/audio/xyz789.wav": "s3"
}
```

## Configuration

### JWT Settings

```bash
# .env
JWT_SECRET_KEY=your-secret-key-here
JWT_COOKIE_NAME=user_session
JWT_COOKIE_MAX_AGE=86400  # 24 hours (session lifetime)
JWT_REFRESH_THRESHOLD=0.5  # Refresh when 50% of lifetime remains
```

### Redis Cleanup Settings

```bash
# .env
TEMP_REDIS_CLEANUP_INTERVAL_SECONDS=300  # Run cleanup every 5 minutes
TEMP_CLEANER_BATCH_SIZE=100  # Max files per cleanup cycle
```

## File Lifecycle Example

### Scenario: User generates audio, session expires

```
Time 0:00:00 - User requests TTS generation
├─► JWT middleware creates session (expires at 24:00:00)
├─► Session tracked in Redis: sessions[user-123] = 24:00:00
├─► Audio file created: /tmp/audio_abc.wav
├─► File registered:
    ├─► files[/tmp/audio_abc.wav] = 0:30:00 (30 min TTL)
    ├─► ownership[/tmp/audio_abc.wav] = user-123
    ├─► storage_type[/tmp/audio_abc.wav] = local
    └─► user_files:user-123.add(/tmp/audio_abc.wav)

Time 0:05:00 - User downloads audio
├─► File ownership verified (user-123 owns file)
├─► File sent to user
└─► File registration removed from Redis (but file still exists)

Time 24:00:00 - Session expires
├─► Cleanup loop detects expired session
├─► Gets all files for user-123
├─► Deletes /tmp/audio_abc.wav from filesystem
├─► Removes session record
└─► Logs: "Cleaned up expired session user-123: 1 files deleted"
```

## Code References

### Middleware
- **File**: `api/src/core/middleware.py`
- **Function**: `JWTCookieMiddleware.dispatch()`
- **Action**: Tracks session expiry via `track_user_session()`

### Temp Manager
- **File**: `api/src/services/temp_manager.py`
- **Functions**:
  - `track_user_session()` - Track session expiry timestamp
  - `register_temp_file()` - Register file with user ownership
  - `remove_temp_registration()` - Remove file from all tracking structures
  - `cleanup_expired_sessions()` - Delete all files for expired sessions
  - `redis_periodic_cleanup_loop()` - Main cleanup loop

## Benefits

### Security
- **Prevents unauthorized access**: Files are deleted when session expires
- **No orphaned data**: All user files are cleaned up together
- **Automatic enforcement**: No manual intervention needed

### Storage Efficiency
- **Immediate cleanup**: Files removed as soon as session expires
- **No accumulation**: Old sessions don't leave files behind
- **S3 integration**: Works with both local and S3 storage

### User Privacy
- **Data lifecycle**: Files exist only during active session
- **Clean slate**: New session = no access to old files
- **Compliance**: Helps meet data retention policies

## Monitoring

### Logs to Watch

```
# Session tracking
DEBUG: Tracked session for user abc-123 (expires at 1704067200.0)

# Session cleanup
INFO: Cleaned up expired session abc-123: 5 files deleted
INFO: Deleted file for expired session abc-123: /tmp/audio_xyz.wav
INFO: Deleted S3 file for expired session abc-123: s3/audio/abc.wav

# Periodic cleanup
INFO: Session cleanup deleted 5 files from expired sessions
```

### Redis Inspection

```bash
# Check sessions
redis-cli ZRANGE temp_files:sessions 0 -1 WITHSCORES

# Check user's files
redis-cli SMEMBERS temp_files:files:user_files:{user_id}

# Check ownership
redis-cli HGETALL temp_files:files:ownership
```

## Troubleshooting

### Issue: Files not being cleaned up

**Check**:
1. Is Redis cleanup loop running? Check startup logs for "Starting Redis temp file cleanup loop"
2. Are sessions being tracked? Check Redis key `temp_files:sessions`
3. Are files registered with user_id? Check Redis hash `temp_files:files:ownership`

**Solutions**:
- Enable Redis logging: Set `TEMP_REDIS_CLEANUP_INTERVAL_SECONDS` to a lower value (e.g., 60)
- Check Redis connection: Verify `REDIS_URL` is correct
- Verify JWT middleware: Ensure `JWTCookieMiddleware` is registered in `main.py`

### Issue: Files deleted too early

**Check**:
1. Is session expiry timestamp correct? Check JWT token expiry (`JWT_COOKIE_MAX_AGE`)
2. Is system time correct? Session expiry uses UTC timestamps

**Solutions**:
- Increase session lifetime: Set `JWT_COOKIE_MAX_AGE` to a higher value
- Verify system clock: Ensure server time is synced with NTP

### Issue: S3 files not being deleted

**Check**:
1. Is S3 client initialized? Check startup logs for "S3 storage enabled"
2. Are files registered as S3? Check Redis hash `temp_files:files:storage_type`
3. Do S3 credentials have delete permissions?

**Solutions**:
- Verify S3 config: Check `S3_ACCESS_KEY`, `S3_ACCESS_SECRET`, `S3_BUCKET_NAME`
- Test S3 deletion manually: Use `delete_from_s3()` function
- Check IAM permissions: Ensure S3 key has `s3:DeleteObject` permission

## Future Enhancements

1. **Configurable session cleanup**:
   - Add setting for aggressive vs conservative cleanup
   - Option to preserve files beyond session expiry

2. **User notifications**:
   - Warn users before session expires
   - Send cleanup summary after session ends

3. **Analytics**:
   - Track cleanup statistics (files deleted, storage freed)
   - User session duration analysis

4. **Grace period**:
   - Keep files for X minutes after session expires
   - Allow session extension for active downloads
