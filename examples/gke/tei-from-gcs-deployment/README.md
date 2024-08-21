# Deploy BGE Base v1.5 (English) with Text Embeddings Inference (TEI) in GKE from a GCS Bucket

BGE, standing for BAAI General Embedding, is a collection of embedding models released by BAAI, which is an English base model for general embedding tasks ranked in the MTEB Leaderboard. Text Embeddings Inference (TEI) is a toolkit developed by Hugging Face for deploying and serving open source text embeddings and sequence classification models; enabling high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5. And, Google Kubernetes Engine (GKE) is a fully-managed Kubernetes service in Google Cloud that can be used to deploy and operate containerized applications at scale using GCP's infrastructure. This post explains how to deploy a text embedding model from a Google Cloud Storage (GCS) Bucket in a GKE Cluster running a purpose-built container to deploy text embedding models in a secure and managed environment with the Hugging Face DLC for TEI.

## Setup / Configuration

First, you need to install both `gcloud` and `kubectl` in your local machine, which are the command-line tools for Google Cloud and Kubernetes, respectively, to interact with the GCP and the GKE Cluster.

* To install `gcloud`, follow the instructions at [Cloud SDK Documentation - Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).
* To install `kubectl`, follow the instructions at [Kubernetes Documentation - Install Tools](https://kubernetes.io/docs/tasks/tools/#kubectl).

Optionally, to ease the usage of the commands within this tutorial, you need to set the following environment variables for GCP:

```bash
export PROJECT_ID="your-project-id"
export LOCATION="your-location"
export CLUSTER_NAME="your-cluster-name"
```

Then you need to login into your GCP account and set the project ID to the one you want to use for the deployment of the GKE Cluster.

```bash
gcloud auth login
gcloud config set project $PROJECT_ID
```

Once you are logged in, you need to enable the necessary service APIs in GCP, such as the Google Kubernetes Engine API, the Google Container Registry API, and the Google Container File System API, which are necessary for the deployment of the GKE Cluster and the Hugging Face DLC for TEI.

```bash
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable containerfilesystem.googleapis.com
```

Additionally, to use `kubectl` with the GKE Cluster credentials, you also need to install the `gke-gcloud-auth-plugin`, that can be installed with `gcloud` as follows:

```bash
gcloud components install gke-gcloud-auth-plugin
```

> [!NOTE]
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit <https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin>.
>
## Create GKE Cluster

Once everything's set up, you can proceed with the creation of the GKE Cluster and the node pool, which in this case will be a single CPU node as for most of the workloads CPU inference is enough to serve most of the text embeddings models, while it could benefit a lot from GPU serving.

> [!NOTE]
> CPU is being used to run the inference on top of the text embeddings models to showcase the current capabilities of TEI, but switching to GPU is as easy as replacing `spec.containers[0].image` with `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-2.ubuntu2204`, and then updating the requested resources, as well as the `nodeSelector` requirements in the `deployment.yaml` file. For more information, please refer to the [`gpu-config`](./gpu-config/) directory that contains a pre-defined configuration for GPU serving in TEI with an NVIDIA Tesla T4 GPU (with a compute capability of 7.5 i.e. natively supported in TEI).

To deploy the GKE Cluster, the "Autopilot" mode will be used as it is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, you can also use the "Standard" mode.

> [!NOTE]
> Important to check before creating the GKE Autopilot Cluster the [GKE Documentation - Optimize Autopilot Pod performance by choosing a machine series](https://cloud.google.com/kubernetes-engine/docs/how-to/performance-pods), since not all the cluster versions support every CPU. Same applies for the GPU support e.g. `nvidia-l4` is not supported in the GKE cluster versions 1.28.3 or lower.

```bash
gcloud container clusters create-auto $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --release-channel=stable \
    --cluster-version=1.28
```

> [!NOTE]
> To select the specific version in your location of the GKE Cluster, you can run the following command:
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

![GKE Cluster in the GCP Console](./imgs/gke-cluster.png)

Once the GKE Cluster is created, you can get the credentials to access it via `kubectl` with the following command:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

## Optional: Upload a model from the Hugging Face Hub to GCS

This is an optional step in the tutorial, since you may want to reuse an existing model on a GCS Bucket, if that's the case, then feel free to jump to the next step of the tutorial on how to configure the IAM for GCS so that you can access the bucket from a pod in the GKE Cluster.

Otherwise, to upload a model from the Hugging Face Hub to a GCS Bucket, you can use the script [./scripts/upload_model_to_gcs.sh](./scripts/upload_model_to_gcs.sh), which will download the model from the Hugging Face Hub and upload it to the GCS Bucket (and create the bucket if not created already).

For convenience, as the reference to the bucket will be used within the following steps, the environment variable `BUCKET_NAME` will be set.

```bash
export BUCKET_NAME="hf-models-gke-bucket"
```

Also, as mentioned above, the `gsutil` component should be installed via `gcloud`, and the Python package `crcmod` should ideally be installed too in order to speed up the upload process via `gsutil cp`.

```bash
gcloud components install gsutil
pip install crcmod
```

Then, you can run the script to download the model from the Hugging Face Hub and then upload it to the GCS Bucket:

> [!NOTE]
> Make sure to set the proper permissions to run the script i.e. `chmod +x ./scripts/upload_model_to_gcs.sh`.

```bash
./scripts/upload_model_to_gcs.sh --model-id BAAI/bge-base-en-v1.5 --gcs gs://$BUCKET_NAME/bge-base-en-v1.5
```

![GCS Bucket in the GCP Console](./imgs/gcs-bucket.png)

## Configure IAM for GCS

Before you proceed with the deployment of the Hugging Face DLC for TEI in the GKE Cluster, you need to set the IAM permissions for the GCS Bucket so that the pod in the GKE Cluster can access the bucket. To do so, you need to create a namespace and a service account in the GKE Cluster, and then set the IAM permissions for the GCS Bucket that contains the model, either as uploaded from the Hugging Face Hub or as already existing in the GCS Bucket.

For convenience, as the reference to both the namespace and the service account will be used within the following steps, the environment variables `NAMESPACE` and `SERVICE_ACCOUNT` will be set.

```bash
export NAMESPACE="hf-gke-namespace"
export SERVICE_ACCOUNT="hf-gke-service-account"
```

Then you can create the namespace and the service account in the GKE Cluster, enabling the creation of the IAM permissions for the pods in that namespace to access the GCS Bucket when using that service account.

```bash
kubectl create namespace $NAMESPACE
kubectl create serviceaccount $SERVICE_ACCOUNT --namespace $NAMESPACE
```

Then you need to add the IAM policy binding to the bucket as follows:

```bash
gcloud storage buckets add-iam-policy-binding \
    gs://$BUCKET_NAME \
    --member "principal://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")/locations/global/workloadIdentityPools/$PROJECT_ID.svc.id.goog/subject/ns/$NAMESPACE/sa/$SERVICE_ACCOUNT" \
    --role "roles/storage.objectUser"
```

## Deploy TEI

Now you can proceed to the Kubernetes deployment of the Hugging Face DLC for TEI, serving the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) model, from a volume mounted in `/data`, copied from the GCS Bucket where the model is located.

> [!NOTE]
> Recently, the Hugging Face Hub team has included the `text-embeddings-inference` tag in the Hub, so feel free to explore all the embedding models in the Hub that can be served via TEI at <https://huggingface.co/models?other=text-embeddings-inference>.

The Hugging Face DLC for TEI will be deployed via `kubectl`, from the configuration files in either the `cpu-config/` or the `gpu-config/` directories depending on whether you want to use the CPU or GPU accelerators, respectively:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face DLC for TEI setting the `MODEL_ID` to the model path in the volume mount, in this case `/data/bge-base-en-v1.5`.
* `service.yaml`: contains the service details of the pod, exposing the port 80 for the TEI service.
* `storageclass.yaml`: contains the storage class details of the pod, defining the storage class for the volume mount.
* (optional) `ingress.yaml`: contains the ingress details of the pod, exposing the service to the external world so that it can be accessed via the ingress IP.

```bash
kubectl apply -f cpu-config/
```

> [!NOTE]
> As already mentioned, for this example you will be deploying the container in a CPU node, but the configuration to deploy TEI in a GPU node is also available in the [`gpu-config`](./gpu-config/) directory, so if you want to deploy TEI in a GPU node, please run `kubectl apply -f gpu-config/` instead of `kubectl apply -f cpu-config/`.

![GKE Deployment in the GCP Console](./imgs/gke-deployment.png)

> [!NOTE]
> The Kubernetes deployment may take a few minutes to be ready, so you can check the status of the deployment with the following command:
>
> ```bash
> kubectl get pods --namespace $NAMESPACE

> ```
>
> Alternatively, you can just wait for the deployment to be ready with the following command:
>
> ```bash
> kubectl wait --for=condition=Available --timeout=700s --namespace $NAMESPACE deployment/tei-deployment
> ```

## Inference with TEI

To run the inference over the deployed TEI service, you can either:

* Port-forwarding the deployed TEI service to the port 8080, so as to access via `localhost` with the command:

    ```bash
    kubectl port-forward --namespace $NAMESPACE service/tei-service 8080:8080
    ```

* Accessing the TEI service via the external IP of the ingress, which is the default scenario here since you have defined the ingress configuration in either the `cpu-configs/ingress.yaml` or the `gpu-config/ingress.yaml` file (but it can be skipped in favour of the port-forwarding), that can be retrieved with the following command:

    ```bash
    kubectl get ingress --namespace $NAMESPACE tei-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    ```

> [!NOTE]
> TEI exposes different inference endpoints based on the task that the model is serving:
>
> * **Text Embeddings**: text embedding models expose the endpoint `/embed` expecting a payload with the key `inputs` which is either a string or a list of strings to be embedded.
> * **Re-rank**: re-ranker models expose the endpoint `/rerank` expecting a payload with the keys `query` and `texts`, where the `query` is the reference used to rank the similarity against each text in `texts`.
> * **Sequence Classification**: classic sequence classification models expose the endpoint `/predict` which expects a payload with the key `inputs` which is either a string or a list of strings to classify.
> More information at <https://huggingface.co/docs/text-embeddings-inference/quick_tour>.

### Via cURL

To send a POST request to the TEI service using `cURL`, you can run the following command:

```bash
curl http://localhost:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Or to send the POST request to the ingress IP:

```bash
curl http://$(kubectl get ingress tei-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Which produces the following output (truncated for brevity, but original tensor length is 768, which is the embedding dimension of [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) i.e. the model you are serving):

```bash
[[-0.01483098,0.010846359,-0.024679236,0.012507628,0.034231555,...]]
```

## Delete GKE Cluster

Finally, once you are done using TEI in the GKE Cluster, you can safely delete the GKE Cluster to avoid incurring in unnecessary costs.

```bash
gcloud container clusters delete $CLUSTER_NAME --location=$LOCATION
```

Alternatively, you can also downscale the replicas of the deployed pod to 0 in case you want to preserve the cluster, since the default GKE Cluster deployed with GKE Autopilot mode is running just a single `e2-small` instance.

```bash
kubectl scale --replicas=0 deployment/tei-deployment
```
