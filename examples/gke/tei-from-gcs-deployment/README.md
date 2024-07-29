# Deploy BGE Base v1.5 (English) with Text Embeddings Inference (TEI) in GKE from a GCS Bucket

TL; DR BGE, standing for BAAI General Embedding, is a collection of embedding models released by BAAI, which is an English base model for general embedding tasks ranked in the MTEB Leaderboard. Text Embeddings Inference (TEI) is a toolkit developed by Hugging Face for deploying and serving open source text embeddings and sequence classification models; enabling high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5. And, Google Kubernetes Engine (GKE) is a fully-managed Kubernetes service in Google Cloud that can be used to deploy and operate containerized applications at scale using GCP's infrastructure. This post explains how to deploy a text embedding model from a Google Cloud Storage (GCS) Bucket in a GKE Cluster running a purpose-built container to deploy text embedding models in a secure and managed environment with the Hugging Face DLC for TEI.

## Setup / Configuration

First, we need to install both `gcloud` and `kubectl` in our local machine, which are the command-line tools for Google Cloud and Kubernetes, respectively, to interact with the GCP and the GKE Cluster.

* To install `gcloud`, follow the instructions at <https://cloud.google.com/sdk/docs/install>.
* To install `kubectl`, follow the instructions at <https://kubernetes.io/docs/tasks/tools/#kubectl>.

Optionally, to ease the usage of the commands within this tutorial, we'll set the following environment variables for GCP:

```bash
export PROJECT_ID="your-project-id"
export LOCATION="your-location"
export CLUSTER_NAME="your-cluster-name"
```

> [!NOTE]
> You may be used to using `REGION` and `ZONE` in GCP, but in this case we will use `LOCATION` instead, which is essentially the same, but it's now the recommended way to refer to the location of the resources in GKE.

Then we need to login into our GCP account and set the project ID to the one we want to use for the deployment of the GKE Cluster.

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
```

Once we are logged in, we need to enable the necessary services in GCP, such as the Google Kubernetes Engine API, the Google Container Registry API, and the Google Container File System API, which are necessary for the deployment of the GKE Cluster and the Hugging Face DLC for TEI.

```bash
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable containerfilesystem.googleapis.com
```

Additionally, in order to use `kubectl` with the GKE Cluster credentials, we also need to install the `gke-gcloud-auth-plugin`, that can be installed with `gcloud` as follows:

```bash
gcloud components install gke-gcloud-auth-plugin
```

> [!NOTE]
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit <https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin>.

## Create GKE Cluster

Once we've set everything up, we are ready to start with the creation of the GKE Cluster and the node pool, which in this case will be a single CPU node as for most of the workloads CPU inference is enough to serve most of the text embeddings models, while it could benefit a lot from GPU serving.

> [!NOTE]
> CPU is being used to run the inference on top of the text embeddings models to showcase the current capabilities of TEI, but switching to GPU is as easy as replacing `spec.containers[0].image` with `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-2.ubuntu2204`, and then updating the requested resources as well as the `nodeSelector` requirements in the `deployment.yaml` file. For more information, please refer to the [`gpu-config`](./gpu-config/) directory that contains a pre-defined configuration for GPU serving in TEI with an NVIDIA Tesla T4 GPU (with a compute capability of 7.5 i.e. natively supported in TEI).

In order to deploy the GKE Cluster, we will use the "Autopilot" mode, which is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, one can also use the "Standard" mode.

> [!NOTE]
> Important to check before creating the GKE Autopilot Cluster <https://cloud.google.com/kubernetes-engine/docs/how-to/performance-pods>, since not all the versions support CPUs, and the CPUs vary depending on the workload. Same applies for the GPU support e.g. `nvidia-l4` is not supported in the GKE cluster versions 1.28.3 or lower.

```bash
gcloud container clusters create-auto $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --release-channel=stable \
    --cluster-version=1.28
```

> [!NOTE]
> To select the specific version in our location of the GKE Cluster, we can run the following command:
>
> ```bash
> gcloud container get-server-config \
>     --flatten="channels" \
>     --filter="channels.channel=STABLE" \
>     --format="yaml(channels.channel,channels.defaultVersion)" \
>     --location=$LOCATION
> ```
>
> For more information please visit <https://cloud.google.com/kubernetes-engine/versioning#specifying_cluster_version>.

As of the GKE documentation and service page in GCP, the creation of the GKE Cluster can take 5 minutes or more, depending on the configuration and the location of the cluster.

![GKE Cluster in the GCP Console](./imgs/gke-cluster.png)

## Optional: Upload a model from the Hugging Face Hub to GCS

This is an optional step in the tutorial, since you may want to reuse an existing model on a GCS bucket, if that's the case, then feel free to jump to the next step of the tutorial on how to configure the IAM for GCS so that you can access the bucket from a pod in the GKE Cluster.

Otherwise, to upload a model from the Hugging Face Hub to a GCS bucket, we can use the script [./scripts/upload_model_to_gcs.sh](./scripts/upload_model_to_gcs.sh), which will download the model from the Hugging Face Hub and upload it to the GCS bucket (and create the bucket if not created already).

Since we will be referring to the GCS bucket name in the upcoming steps, we will set the environment variable `BUCKET_NAME` to the name of the bucket we want to use for the deployment of the model.

```bash
export BUCKET_NAME="hf-models-gke-bucket"
```

Also, as mentioned above, the `gsutil` component should be installed via `gcloud`, and the Python package `crcmod` should ideally be installed too in order to speed up the upload process via `gsutil cp`.

```bash
gcloud components install gsutil
pip install crcmod
```

Then, we can run the script to upload the model to the GCS bucket:

> [!NOTE]
> Make sure to set the proper permissions to run the script i.e. `chmod +x ./scripts/upload_model_to_gcs.sh`.

```bash
./scripts/upload_model_to_gcs.sh --model-id BAAI/bge-base-en-v1.5 --gcs gs://$BUCKET_NAME/bge-base-en-v1.5
```

![GCS Bucket in the GCP Console](./imgs/gcs-bucket.png)

## Configure IAM for GCS

Before we proceed with the deployment of the Hugging Face LLM DLC for TEI in the GKE Cluster, we need to set the IAM permissions for the GCS bucket so that the pod in the GKE Cluster can access the bucket. To do so, we will create a namespace and a service account in the GKE Cluster, and then set the IAM permissions for the GCS bucket that contains the model, either as uploaded from the Hugging Face Hub or as already existing in the GCS bucket.

In order to set the Kubernetes secret, we first need to get the credentials of the GKE Cluster so that we can access it via `kubectl`:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

> [!NOTE]
> The `gcloud container clusters get-credentials` command will set the `kubectl` context to the GKE Cluster, so that we can interact with the cluster via `kubectl`, meaning it will be required for the rest of the tutorial, but only needs to be ran once, that's why in the following steps we will not include it in the commands as we're assuming it's already set.

Since we will be referring to the namespace and the service account in the upcoming steps, we will set the environment variables `NAMESPACE` and `SERVICE_ACCOUNT` to the name of the namespace and the service account we want to use for the deployment of the model.

```bash
export NAMESPACE="hf-gke-namespace"
export SERVICE_ACCOUNT="hf-gke-service-account"
```

Then we can create the namespace and the service account in the GKE Cluster, so that we can then create the IAM permissions for the pods in that namespace to access the GCS bucket with the model when using that service account.

```bash
kubectl create namespace $NAMESPACE
kubectl create serviceaccount $SERVICE_ACCOUNT --namespace $NAMESPACE
```

Then we add the IAM policy binding to the bucket as follows:

```bash
gcloud storage buckets add-iam-policy-binding \
    gs://$BUCKET_NAME \
    --member "principal://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")/locations/global/workloadIdentityPools/$PROJECT_ID.svc.id.goog/subject/ns/$NAMESPACE/sa/$SERVICE_ACCOUNT" \
    --role "roles/storage.objectUser"
```

## Deploy TEI

Once we are all set up, we can proceed to the Kubernetes deployment of the Hugging Face LLM DLC for TEI, serving the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) model from a volume mount under `/data` copied from the GCS bucket where the model is.

Then we can already deploy the Hugging Face LLM DLC for TEI via `kubectl`, from the following configuration files in either the `cpu-config/` or the `gpu-config/` directories:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face LLM DLC setting the `MODEL_ID` to the model path in the volume mount, in this case `/data/bge-base-en-v1.5`.
* `service.yaml`: contains the service details of the pod, exposing the port 80 for the TEI service.
* (optional) `ingress.yaml`: contains the ingress details of the pod, exposing the service to the external world so that it can be accessed via the ingress IP.

> [!NOTE]
> As already mentioned, for this example we will be deploying the container in a CPU node, but the configuration to deploy TEI in a GPU node is also available in the [`gpu-config`](./gpu-config/) directory, so if you're willing to deploy TEI in a GPU node, please run `kubectl apply -f gpu-config/` instead of `kubectl apply -f cpu-config/` in the following command.

```bash
kubectl apply -f cpu-config/
```

![GKE Deployment in the GCP Console](./imgs/gke-deployment.png)

> [!NOTE]
> The Kubernetes deployment may take a few minutes to be ready, so we can check the status of the deployment with the following command:
>
> ```bash
> kubectl get pods --namespace $NAMESPACE
> ```
>
> Alternatively, we can just wait for the deployment to be ready with the following command:
>
> ```bash
> kubectl wait --for=condition=Available --timeout=700s --namespace=$NAMESPACE deployment/tei-deployment
> ```

## Inference with TEI

In order to run the inference over the deployed TEI service, we can either:

* Port-forwarding the deployed TEI service to the port 8080, so as to access via `localhost` with the command:

    ```bash
    kubectl port-forward --namespace $NAMESPACE service/tei-service 8080:8080
    ```

* Accessing the TEI service via the external IP of the ingress, which is the default scenario here since we have defined the ingress configuration in either the `cpu-configs/ingress.yaml` or the `gpu-config/ingress.yaml` file (but it can be skipped in favour of the port-forwarding), that can be retrieved with the following command:

    ```bash
    kubectl get ingress --namespace $NAMESPACE tei-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    ```

> [!NOTE]
> TEI exposes different inference endpoints based on the task that the model is serving:
>
> * Text Embeddings: text embedding models expose the endpoint `/embed` expecting a payload with the key `inputs` which is either a string or a list of strings to be embedded.
> * Re-rank: re-ranker models expose the endpoint `/rerank` expecting a payload with the keys `query` and `texts`, where the `query` is the reference used to rank the similarity against each text in `texts`.
> * Sequence Classification: classic sequence classification models expose the endpoint `/predict` which expects a payload with the key `inputs` which is either a string or a list of strings to classify.
> More information at <https://huggingface.co/docs/text-embeddings-inference/quick_tour>.

### Via cURL

To send a POST request to the TEI service using `cURL`, we can run the following command:

```bash
curl http://localhost:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Or to send the POST request to the ingress IP:

```bash
curl http://<ingress-ip>/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Which produces the following output (truncated for brevity, but original tensor length is 768, which is the embedding dimension of [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) i.e. the model we're serving):

```bash
[[-0.01483098,0.010846359,-0.024679236,0.012507628,0.034231555,...]]⏎
```

## Delete GKE Cluster

Finally, once we are done using TEI in the GKE Cluster, we can safely delete the cluster we've just created to avoid incurring in unnecessary costs.

```bash
gcloud container clusters delete $CLUSTER_NAME --location=$LOCATION
```

Alternatively, we can also downscale the replicas of the deployed pod to 0 in case we want to preserve the cluster, since the default GKE Cluster deployed with GKE Autopilot mode is running just a single `e2-small` instance.

```bash
kubectl scale --replicas=0 deployment/tei-deployment
```
