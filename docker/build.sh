#!/bin/bash
set -e

# Get version from argument or use default
VERSION=${1:-"latest"}

# GitHub Container Registry settings
REGISTRY="ghcr.io"
OWNER="remsky"
REPO="kokoro-fastapi"

# Create and use a new builder that supports multi-platform builds
docker buildx create --name multiplatform-builder --use || true

# Enable BuildKit garbage collection
export BUILDKIT_STEP_LOG_MAX_SIZE=10485760
export DOCKER_BUILDKIT=1

# Build CPU image with multi-platform support
echo "Building CPU image..."
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION} \
  -t ${REGISTRY}/${OWNER}/${REPO}-cpu:latest \
  -f docker/cpu/Dockerfile \
  --push \
  --build-arg BUILDKIT_INLINE_CACHE=1 .

# Clean up between builds
docker builder prune -f

# Build GPU image with multi-platform support
echo "Building GPU image..."
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION} \
  -t ${REGISTRY}/${OWNER}/${REPO}-gpu:latest \
  -f docker/gpu/Dockerfile \
  --push \
  --build-arg BUILDKIT_INLINE_CACHE=1 .

echo "Build complete!"
echo "Created images:"
echo "- ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION} (linux/amd64, linux/arm64)"
echo "- ${REGISTRY}/${OWNER}/${REPO}-cpu:latest (linux/amd64, linux/arm64)"
echo "- ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION} (linux/amd64, linux/arm64)"
echo "- ${REGISTRY}/${OWNER}/${REPO}-gpu:latest (linux/amd64, linux/arm64)"
