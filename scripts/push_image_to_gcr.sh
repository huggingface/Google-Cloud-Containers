#!/bin/bash

# This script pushes a built Docker image into the Artifact Registry in GCP.
# ./push_image_to_gcr.sh --image my-image --project-id my-project --location us-central1 --repository deep-learning-images --destination us-central1-docker.pkg.dev/my-project/deep-learning-images/huggingface-image

# Parse command-line arguments for repository and bucket
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --image)
        IMAGE="$2"
        shift
        ;;
    --project-id)
        PROJECT_ID="$2"
        shift
        ;;
    --location)
        LOCATION="$2"
        shift
        ;;
    --repository)
        REPOSITORY="$2"
        shift
        ;;
    --destination)
        DESTINATION="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

# Check if necessary parameters are provided
if [[ -z "$IMAGE" || -z "$PROJECT_ID" || -z "$LOCATION" || -z "$REPOSITORY" || -z "$DESTINATION" ]]; then
    echo "Usage: $0 --image my-image --project-id gcs-project --location us-central1 --repository my-repo --destination my-destination-image"
    exit 1
fi

# Authenticate with GCP
gcloud auth application-default login
gcloud config set project $PROJECT_ID

# Enable the Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# Check whether the repository already exists, if not, create it
echo "Checking if the repository $REPOSITORY exists..."
gcloud artifacts repositories describe $REPOSITORY --location $LOCATION
if [ $? -ne 0 ]; then
    echo "Repository $REPOSITORY does not exist, creating it..."
    gcloud artifacts repositories create $REPOSITORY --repository-format=docker --location $LOCATION
    if [ $? -ne 0 ]; then
        echo "Failed to create repository $REPOSITORY with error code $?"
        exit 1
    fi
    echo "Repository $REPOSITORY successfully created!"
else
    REPOSITORY_FORMAT=$(gcloud artifacts repositories describe $REPOSITORY --location $LOCATION --format="value(format)")
    if [ "$REPOSITORY_FORMAT" != "DOCKER" ]; then
        echo "Repository $REPOSITORY already exists but is not a Docker repository!"
        exit 1
    fi
    echo "Repository $REPOSITORY already exists!"
fi

# Configure Docker to use the Artifact Registry repository just created
gcloud auth configure-docker $LOCATION-docker.pkg.dev

# Finally, we tag our image with the target / destination name, and then we push it to the Artifact Registry repository
echo "Pushing image $IMAGE to $DESTINATION..."
docker tag $IMAGE $DESTINATION
docker push $DESTINATION
echo "Docker image $IMAGE successfully pushed to $DESTINATION!"
