#!/bin/bash

# Build the Docker image if needed
if [[ "$(docker images -q kokoro-locust 2> /dev/null)" == "" ]]; then
    echo "Building Kokoro Locust image..."
    docker build -t kokoro-locust .
fi

# Array of test configurations: instances,users,spawn_rate,run_time
configs=(
    "1,5,1,3m"     # Light load: 1 instance, 5 users
    "2,10,2,3m"    # Medium load: 2 instances, 10 users
    "4,20,2,3m"    # Heavy load: 4 instances, 20 users
)

# Create results directory
mkdir -p test_results
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="test_results/run_${timestamp}"
mkdir -p "$results_dir"

# Run tests for each configuration
for config in "${configs[@]}"; do
    IFS=',' read -r instances users spawn_rate run_time <<< "$config"
    
    echo "----------------------------------------"
    echo "Testing with configuration:"
    echo "- Model instances: $instances"
    echo "- Concurrent users: $users"
    echo "- Spawn rate: $spawn_rate"
    echo "- Run time: $run_time"
    echo "----------------------------------------"
    
    # Export instance count for the server (if running locally)
    export PYTORCH_MAX_CONCURRENT_MODELS=$instances
    
    # Run load test
    docker run --rm \
        -e LOCUST_HOST=http://host.docker.internal:8880 \
        -e LOCUST_HEADLESS=true \
        -e LOCUST_USERS=$users \
        -e LOCUST_SPAWN_RATE=$spawn_rate \
        -e LOCUST_RUN_TIME=$run_time \
        --name kokoro-locust-test \
        kokoro-locust
    
    # Copy the report
    test_name="instances${instances}_users${users}"
    docker cp kokoro-locust-test:/locust/report.html "$results_dir/${test_name}_report.html"
    
    echo "Test complete. Report saved to $results_dir/${test_name}_report.html"
    echo "Waiting 30s before next test..."
    sleep 30
done

echo "----------------------------------------"
echo "All tests complete!"
echo "Results saved in: $results_dir"
echo "----------------------------------------"