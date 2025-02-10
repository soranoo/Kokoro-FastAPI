#!/bin/bash

# If LOCUST_HEADLESS is true, run in headless mode with specified parameters
if [ "$LOCUST_HEADLESS" = "true" ]; then
    locust -f locustfile.py \
        --host ${LOCUST_HOST} \
        --users ${LOCUST_USERS} \
        --spawn-rate ${LOCUST_SPAWN_RATE} \
        --run-time ${LOCUST_RUN_TIME} \
        --headless \
        --print-stats \
        --html report.html
else
    # Run with web interface
    locust -f locustfile.py --host ${LOCUST_HOST}
fi