import logging
import os
import threading
import time

import docker
import pytest
import requests

from docker.types.containers import DeviceRequest

from ..utils import gpu_available, stream_logs, supports_flash_attention

MAX_RETRIES = 10


@pytest.mark.skipif(not gpu_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "text_generation_launcher_kwargs",
    [
        {
            "MODEL_ID": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "MAX_INPUT_TOKENS": "512",
            "MAX_TOTAL_TOKENS": "1024",
            "MAX_BATCH_PREFILL_TOKENS": "1512",
        },
        {
            "MODEL_ID": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "MAX_INPUT_TOKENS": "512",
            "MAX_TOTAL_TOKENS": "1024",
            "MAX_BATCH_PREFILL_TOKENS": "1512",
            "AIP_MODE": "PREDICTION",
        },
    ],
)
def test_text_generation_inference(
    caplog: pytest.LogCaptureFixture,
    text_generation_launcher_kwargs: dict,
) -> None:
    caplog.set_level(logging.INFO)

    container_uri = os.getenv("TGI_DLC", None)
    if container_uri is None or container_uri == "":
        assert False, "TGI_DLC environment variable is not set"

    client = docker.from_env()

    # If the GPU doesn't support Flash Attention, then set `USE_FLASH_ATTENTION=false`
    if not supports_flash_attention():
        text_generation_launcher_kwargs["USE_FLASH_ATTENTION"] = "false"

    logging.info(
        f"Starting container for {text_generation_launcher_kwargs.get('MODEL_ID', None)}..."
    )
    container = client.containers.run(
        container_uri,
        ports={8080: 8080},
        environment=text_generation_launcher_kwargs,
        healthcheck={
            "test": ["CMD", "curl", "-s", "http://localhost:8080/health"],
            "interval": int(30 * 1e9),
            "timeout": int(30 * 1e9),
            "retries": 3,
            "start_period": int(30 * 1e9),
        },
        platform="linux/amd64",
        detach=True,
        # Extra kwargs related to the CUDA devices
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    )
    logging.info(f"Container {container.id} started...")  # type: ignore

    # Start log streaming in a separate thread
    log_thread = threading.Thread(target=stream_logs, args=(container,))
    log_thread.daemon = True
    log_thread.start()

    # Get endpoint names for both health and predict (may differ if AIP env vars are defined)
    health_route = os.getenv("AIP_HEALTH_ROUTE", "/health")
    predict_route = (
        os.getenv("AIP_PREDICT_ROUTE", "/predict")
        if os.getenv("AIP_MODE")
        else "/generate"
    )

    container_healthy = False
    for _ in range(MAX_RETRIES):
        # It the container failed to start properly, then the health check will fail
        if container.status == "exited":  # type: ignore
            container_healthy = False
            break

        try:
            logging.info(
                f"Trying to connect to http://localhost:8080{health_route} [retry {_ + 1}/{MAX_RETRIES}]..."
            )
            response = requests.get(f"http://localhost:8080{health_route}")
            assert response.status_code == 200
            container_healthy = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(30)

    if not container_healthy:
        logging.error("Container is not healthy after several retries...")
        container.stop()  # type: ignore

    assert container_healthy

    container_failed = False
    try:
        for prompt in ["What's Deep Learning?", "What's the capital of France?"]:
            logging.info(
                f"Sending prediction request for {prompt=} to http://localhost:8080{predict_route}..."
            )
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "top_p": 0.95,
                    "temperature": 1.0,
                },
            }

            if os.getenv("AIP_MODE"):
                payload = {"instances": [payload]}

            start_time = time.perf_counter()
            response = requests.post(
                f"http://localhost:8080{predict_route}",
                json=payload,
            )
            end_time = time.perf_counter()

            assert response.status_code in [200, 201]
            assert "generated_text" in response.json()

            logging.info(
                f"Prediction request for {prompt=} took {end_time - start_time:.2f}s"
            )
    except Exception as e:
        logging.error(
            f"Error while sending prediction request with exception: {e}"  # type: ignore
        )
        container_failed = True
    finally:
        if log_thread.is_alive():
            log_thread.join(timeout=5)

        logging.info(f"Stopping container {container.id}...")  # type: ignore
        container.stop()  # type: ignore
        container.remove()  # type: ignore

    assert not container_failed
