# Import necessary libraries
from google.cloud import container_v1
import pytest


@pytest.fixture
def gke_client():
    # Set up GKE client
    return container_v1.ClusterManagerClient()


def test_create_cluster(gke_client):
    # Create a new GKE cluster
    cluster_name = "my-cluster"
    project_id = "my-project"
    zone = "us-central1-a"
    cluster = {
        "name": cluster_name,
        "initial_node_count": 1,
        # Add other cluster configuration parameters here
    }
    operation = gke_client.create_cluster(project_id, zone, cluster)

    # Wait for the operation to complete
    operation.result()

    # Assert that the cluster is created successfully
    cluster = gke_client.get_cluster(project_id, zone, cluster_name)
    assert cluster.status == "RUNNING"


def test_deploy_container(gke_client):
    # Deploy a container on the GKE cluster
    cluster_name = "my-cluster"
    project_id = "my-project"
    zone = "us-central1-a"
    container_name = "my-container"
    image = "gcr.io/my-project/my-image:latest"
    container = {
        "name": container_name,
        "image": image,
        # Add other container configuration parameters here
    }
    operation = gke_client.create_container(project_id, zone, cluster_name, container)

    # Wait for the operation to complete
    operation.result()

    # Assert that the container is deployed successfully
    container = gke_client.get_container(project_id, zone, cluster_name, container_name)
    assert container.status == "RUNNING"
