from time import sleep

import docker
import pytest
import requests


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
    ],
)
def test_transformers(
    hf_model_id: str,
    hf_task: str,
    prediction_payload: dict,
) -> None:
    client = docker.from_env()

    print(f"Starting container for {hf_model_id}...")
    container = client.containers.run(
        "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-2.transformers.4-44.ubuntu2204.py311",
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
        # To show all the `logging` messages from the container
        stdin_open=True,
        tty=True,
    )

    print(f"Container {container.id} started...")  # type: ignore
    for _ in range(MAX_RETRIES):
        try:
            print(
                f"Trying to connect to http://localhost:8080/health [retry {_ + 1}/{MAX_RETRIES}]..."
            )
            response = requests.get("http://localhost:8080/health")
            assert response.status_code == 200
            break
        except requests.exceptions.ConnectionError:
            sleep(10)

    try:
        response = requests.post(
            "http://localhost:8080/predict",
            json=prediction_payload,
        )
        assert response.status_code in [200, 201]
        assert "predictions" in response.json()
    finally:
        print(f"Stopping container {container.id}...")  # type: ignore
        container.stop()  # type: ignore
        container.remove()  # type: ignore
