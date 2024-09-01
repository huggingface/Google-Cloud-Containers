import logging
import os
import threading
import time

import docker
import pytest
import requests

from docker.types.containers import DeviceRequest

from ..constants import CUDA_AVAILABLE
from ..utils import stream_logs

MAX_RETRIES = 10


@pytest.mark.parametrize(
    "text_embeddings_router_kwargs",
    [
        {
            "MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "MODEL_ID": "sentence-transformers/all-MiniLM-L6-v2",
            "AIP_MODE": "PREDICTION",
        },
    ],
)
def test_text_embeddings_inference(
    caplog: pytest.LogCaptureFixture,
    text_embeddings_router_kwargs: dict,
) -> None:
    caplog.set_level(logging.INFO)

    client = docker.from_env()

    cuda_kwargs = {}
    if CUDA_AVAILABLE:
        cuda_kwargs = {
            "runtime": "nvidia",
            "device_requests": [DeviceRequest(count=-1, capabilities=[["gpu"]])],
        }

    logging.info(
        f"Starting container for {text_embeddings_router_kwargs.get('MODEL_ID', None)}..."
    )
    container = client.containers.run(
        os.getenv(
            "TEI_DLC",
            "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cpu.1-2"
            if not CUDA_AVAILABLE
            else "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-4.ubuntu2204",
        ),
        # TODO: udpate once the TEI DLCs is updated, as the current is still on revision:
        # https://github.com/huggingface/Google-Cloud-Containers/blob/517b8728725f6249774dcd46ee8d7ede8d95bb70/containers/tei/cpu/1.2.2/Dockerfile
        # and it exposes the 80 port and uses the /data directory instead of /tmp
        ports={8080 if CUDA_AVAILABLE else 80: 8080},
        environment=text_embeddings_router_kwargs,
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
        **cuda_kwargs,
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
        else "/embed"
    )

    container_healthy = False
    for _ in range(MAX_RETRIES):
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
        logging.info(
            f"Sending prediction request to http://localhost:8080{predict_route}..."
        )
        payload = {"inputs": "What's Deep Learning?"}

        if os.getenv("AIP_MODE"):
            payload = {"instances": [payload]}

        start_time = time.perf_counter()
        response = requests.post(
            f"http://localhost:8080{predict_route}",
            json=payload,
        )
        end_time = time.perf_counter()

        assert response.status_code in [200, 201]
        assert response.json() is not None

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
