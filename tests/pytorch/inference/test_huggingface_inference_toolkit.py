import logging
import time

import docker
import pytest
import requests

from docker.types.containers import DeviceRequest

from tests.constants import CUDA_AVAILABLE

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
            "runwayml/stable-diffusion-v1-5",
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
def test_transformers(
    caplog: pytest.LogCaptureFixture,
    hf_model_id: str,
    hf_task: str,
    prediction_payload: dict,
) -> None:
    caplog.set_level(logging.INFO)

    client = docker.from_env()

    cuda_kwargs = {}
    if CUDA_AVAILABLE:
        cuda_kwargs = {
            "runtime": "nvidia",
            "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
        }

    logging.info(f"Starting container for {hf_model_id}...")
    container = client.containers.run(
        "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-2.transformers.4-44.ubuntu2204.py311",
        ports={"8080": 8080},
        environment={
            "HF_MODEL_ID": hf_model_id,
            # "HF_MODEL_DIR": "/opt/huggingface/model",
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
        volumes=[
            f"/Users/alvarobartt/HuggingFace/Google-Cloud-Containers/{hf_task}:/opt/huggingface/model"
        ],
        detach=True,
        # To show all the `logging` messages from the container
        stdin_open=True,
        tty=True,
        # Extra kwargs related to the CUDA devices
        **cuda_kwargs,
    )

    logging.info(f"Container {container.id} started...")  # type: ignore
    container_healthy = False
    for _ in range(MAX_RETRIES):
        try:
            logging.info(
                f"Trying to connect to http://localhost:8080/health [retry {_ + 1}/{MAX_RETRIES}]..."
            )
            response = requests.get("http://localhost:8080/health")
            assert response.status_code == 200
            container_healthy = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(10)

    if not container_healthy:
        logging.error("Container is not healthy after several retries...")
        container.stop()
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
            f"Error while sending prediction request with exception: {e}; and container logs: {container.logs()}"
        )
        container_failed = True
    finally:
        logging.info(f"Stopping container {container.id}...")  # type: ignore
        container.stop()  # type: ignore
        container.remove()  # type: ignore

    assert not container_failed
