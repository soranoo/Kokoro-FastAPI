# Kokoro FastAPI Load Testing

This directory contains load testing scripts using Locust to test the Kokoro FastAPI server's performance under concurrent load.

## Docker Setup

The easiest way to run the tests is using Docker:

```bash
# Build the Docker image
docker build -t kokoro-locust .

# Run with web interface (default)
docker run -p 8089:8089 -e LOCUST_HOST=http://host.docker.internal:8880 kokoro-locust

# Run headless mode with specific parameters
docker run -e LOCUST_HOST=http://host.docker.internal:8880 \
    -e LOCUST_HEADLESS=true \
    -e LOCUST_USERS=10 \
    -e LOCUST_SPAWN_RATE=1 \
    -e LOCUST_RUN_TIME=5m \
    kokoro-locust
```

### Environment Variables

- `LOCUST_HOST`: Target server URL (default: http://localhost:8880)
- `LOCUST_USERS`: Number of users to simulate (default: 10)
- `LOCUST_SPAWN_RATE`: Users to spawn per second (default: 1)
- `LOCUST_RUN_TIME`: Test duration (default: 5m)
- `LOCUST_HEADLESS`: Run without web UI if true (default: false)

### Accessing Results

- Web UI: http://localhost:8089 when running in web mode
- HTML Report: Generated in headless mode, copy from container:
  ```bash
  docker cp <container_id>:/locust/report.html ./report.html
  ```

## Local Setup (Alternative)

If you prefer running without Docker:

1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Make sure your Kokoro FastAPI server is running (default: http://localhost:8880)

3. Run Locust:
```bash
# Web UI mode
locust -f locustfile.py --host http://localhost:8880

# Headless mode
locust -f locustfile.py --host http://localhost:8880 --users 10 --spawn-rate 1 --run-time 5m --headless
```

## Test Scenarios

The load test includes:
1. TTS endpoint testing with short phrases
2. Model pool monitoring

## Testing Different Configurations

To test with different numbers of model instances:

1. Set the model instance count in your server environment:
```bash
export PYTORCH_MAX_CONCURRENT_MODELS=2  # Adjust as needed
```

2. Restart your Kokoro FastAPI server

3. Run the load test with different user counts:
```bash
# Example: Test with 20 users
docker run -e LOCUST_HOST=http://host.docker.internal:8880 \
    -e LOCUST_HEADLESS=true \
    -e LOCUST_USERS=20 \
    -e LOCUST_SPAWN_RATE=2 \
    -e LOCUST_RUN_TIME=5m \
    kokoro-locust
```

## Example Test Matrix

Test your server with different configurations:

| Model Instances | Concurrent Users | Expected Load |
|----------------|------------------|---------------|
| 1              | 5                | Light         |
| 2              | 10               | Medium        |
| 4              | 20               | Heavy         |

## Quick Test Script

Here's a quick script to test multiple configurations:

```bash
#!/bin/bash

# Array of test configurations
configs=(
    "1,5"    # 1 instance, 5 users
    "2,10"   # 2 instances, 10 users
    "4,20"   # 4 instances, 20 users
)

for config in "${configs[@]}"; do
    IFS=',' read -r instances users <<< "$config"
    
    echo "Testing with $instances instances and $users users..."
    
    # Set instance count on server (you'll need to implement this)
    # ssh server "export PYTORCH_MAX_CONCURRENT_MODELS=$instances && restart_server"
    
    # Run load test
    docker run -e LOCUST_HOST=http://host.docker.internal:8880 \
        -e LOCUST_HEADLESS=true \
        -e LOCUST_USERS=$users \
        -e LOCUST_SPAWN_RATE=1 \
        -e LOCUST_RUN_TIME=5m \
        kokoro-locust
    
    echo "Waiting 30s before next test..."
    sleep 30
done
```

## Tips

1. Start with low user counts and gradually increase
2. Monitor server resources during tests
3. Use the debug endpoint (/debug/model_pool) to monitor instance usage
4. Check server logs for any errors or bottlenecks
5. When using Docker, use `host.docker.internal` to access localhost