import logging
import os
import threading
import time

import docker
import pytest
import requests

from docker.types.containers import DeviceRequest

from ...utils import gpu_available, stream_logs

MAX_RETRIES = 10


# Tests below are only on some combinations of models and tasks, since most of those
# tests are already available within https://github.com/huggingface/huggingface-inference-toolkit
# as `huggingface-inference-toolkit` is the inference engine powering the PyTorch DLCs for Inference
@pytest.mark.parametrize(
    ("hf_model_id", "hf_task", "prediction_payload"),
    [
        (
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "text-classification",
            {
                "instances": ["I love this product", "I hate this product"],
                "parameters": {"top_k": 2},
            },
        ),
        (
            "BAAI/bge-base-en-v1.5",
            "sentence-embeddings",
            {"instances": ["I love this product"]},
        ),
        (
            "lambdalabs/miniSD-diffusers",
            "text-to-image",
            {
                "instances": ["A cat holding a sign that says hello world"],
                "parameters": {
                    "negative_prompt": "",
                    "num_inference_steps": 2,
                    "guidance_scale": 0.7,
                },
            },
        ),
    ],
)
def test_huggingface_inference_toolkit(
    caplog: pytest.LogCaptureFixture,
    hf_model_id: str,
    hf_task: str,
    prediction_payload: dict,
) -> None:
    caplog.set_level(logging.INFO)

    container_uri = os.getenv("INFERENCE_DLC", None)
    if container_uri is None or container_uri == "":
        assert False, "INFERENCE_DLC environment variable is not set"

    client = docker.from_env()

    logging.info(f"Starting container for {hf_model_id}...")
    container = client.containers.run(
        container_uri,
        ports={"8080": 8080},
        environment={
            "HF_MODEL_ID": hf_model_id,
            "HF_TASK": hf_task,
            "AIP_MODE": "PREDICTION",
            "AIP_HTTP_PORT": "8080",
            "AIP_PREDICT_ROUTE": "/predict",
            "AIP_HEALTH_ROUTE": "/health",
        },
        healthcheck={
            "test": ["CMD", "curl", "-s", "http://localhost:8080/health"],
            "interval": int(30 * 1e9),
            "timeout": int(30 * 1e9),
            "retries": 3,
            "start_period": int(30 * 1e9),
        },
        platform="linux/amd64",
        detach=True,
        # Extra `device_requests` related to the CUDA devices if any
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
        if gpu_available()
        else None,
    )

    # Start log streaming in a separate thread
    log_thread = threading.Thread(target=stream_logs, args=(container,))
    log_thread.daemon = True
    log_thread.start()

    logging.info(f"Container {container.id} started...")  # type: ignore
    container_healthy = False
    for _ in range(MAX_RETRIES):
        # It the container failed to start properly, then the health check will fail
        if container.status == "exited":  # type: ignore
            container_healthy = False
            break

        try:
            logging.info(
                f"Trying to connect to http://localhost:8080/health [retry {_ + 1}/{MAX_RETRIES}]..."
            )
            response = requests.get("http://localhost:8080/health")
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
        logging.info("Sending prediction request to http://localhost:8080/predict...")
        start_time = time.perf_counter()
        response = requests.post(
            "http://localhost:8080/predict",
            json=prediction_payload,
        )
        end_time = time.perf_counter()
        assert response.status_code in [200, 201]
        assert "predictions" in response.json()
        logging.info(f"Prediction request took {end_time - start_time:.2f}s")
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
