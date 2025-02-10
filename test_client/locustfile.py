from locust import HttpUser, task, between, events
import json
import time

class SystemStats:
    def __init__(self):
        self.queue_size = 0
        self.active_instances = 0
        self.gpu_memory_used = 0
        self.cpu_percent = 0
        self.memory_percent = 0
        self.error_count = 0
        self.last_error = None

system_stats = SystemStats()

@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    @environment.web_ui.app.route("/system-stats")
    def system_stats_page():
        return {
            "queue_size": system_stats.queue_size,
            "active_instances": system_stats.active_instances,
            "gpu_memory_used": system_stats.gpu_memory_used,
            "cpu_percent": system_stats.cpu_percent,
            "memory_percent": system_stats.memory_percent,
            "error_count": system_stats.error_count,
            "last_error": system_stats.last_error
        }

class KokoroUser(HttpUser):
    wait_time = between(2, 3)  # Increased wait time to reduce load

    def on_start(self):
        """Initialize test data."""
        self.test_phrases = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Testing voice synthesis with a short phrase.",
            "I hope this works well!",
            "Just a quick test of the system."
        ]
        
        self.test_config = {
            "model": "kokoro",
            "voice": "af_nova",
            "response_format": "mp3",
            "speed": 1.0,
            "stream": False
        }

    @task(1)
    def test_tts_endpoint(self):
        """Test the TTS endpoint with short phrases."""
        import random
        test_text = random.choice(self.test_phrases)
        
        payload = {
            **self.test_config,
            "input": test_text
        }

        with self.client.post(
            "/v1/audio/speech",
            json=payload,
            catch_response=True,
            name="/v1/audio/speech (short text)"
        ) as response:
            try:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 429:  # Too Many Requests
                    response.failure("Rate limit exceeded")
                    system_stats.error_count += 1
                    system_stats.last_error = "Rate limit exceeded"
                elif response.status_code >= 500:
                    error_msg = f"Server error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'detail' in error_data:
                            error_msg = f"Server error: {error_data['detail']}"
                    except:
                        pass
                    response.failure(error_msg)
                    system_stats.error_count += 1
                    system_stats.last_error = error_msg
                else:
                    response.failure(f"Unexpected status: {response.status_code}")
                    system_stats.error_count += 1
                    system_stats.last_error = f"Unexpected status: {response.status_code}"
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                response.failure(error_msg)
                system_stats.error_count += 1
                system_stats.last_error = error_msg

    @task(1)  # Reduced monitoring frequency
    def monitor_system(self):
        """Monitor system metrics via debug endpoints."""
        # Get model pool stats
        with self.client.get(
            "/debug/model_pool",
            catch_response=True,
            name="Debug - Model Pool"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                system_stats.queue_size = data.get("queue_size", 0)
                system_stats.active_instances = data.get("active_instances", 0)
                if "gpu_memory" in data:
                    system_stats.gpu_memory_used = data["gpu_memory"]["used_mb"]
                
                # Report metrics
                self.environment.events.request.fire(
                    request_type="METRIC",
                    name="Queue Size",
                    response_time=system_stats.queue_size,
                    response_length=0,
                    exception=None
                )
                self.environment.events.request.fire(
                    request_type="METRIC",
                    name="Active Instances",
                    response_time=system_stats.active_instances,
                    response_length=0,
                    exception=None
                )
                if "gpu_memory" in data:
                    self.environment.events.request.fire(
                        request_type="METRIC",
                        name="GPU Memory (MB)",
                        response_time=system_stats.gpu_memory_used,
                        response_length=0,
                        exception=None
                    )

        # Get system stats
        with self.client.get(
            "/debug/system",
            catch_response=True,
            name="Debug - System"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                system_stats.cpu_percent = data.get("cpu", {}).get("cpu_percent", 0)
                system_stats.memory_percent = data.get("process", {}).get("memory_percent", 0)
                
                # Report metrics
                self.environment.events.request.fire(
                    request_type="METRIC",
                    name="CPU %",
                    response_time=system_stats.cpu_percent,
                    response_length=0,
                    exception=None
                )
                self.environment.events.request.fire(
                    request_type="METRIC",
                    name="Memory %",
                    response_time=system_stats.memory_percent,
                    response_length=0,
                    exception=None
                )

# Add custom charts
@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        '--custom-stats',
        dest='custom_stats',
        action='store_true',
        help='Enable custom statistics in web UI'
    )

# Stats processor
def process_stats():
    stats = {
        "Queue Size": system_stats.queue_size,
        "Active Instances": system_stats.active_instances,
        "GPU Memory (MB)": system_stats.gpu_memory_used,
        "CPU %": system_stats.cpu_percent,
        "Memory %": system_stats.memory_percent,
        "Error Count": system_stats.error_count
    }
    return stats

@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    print("\nFinal System Stats:")
    for metric, value in process_stats().items():
        print(f"{metric}: {value}")