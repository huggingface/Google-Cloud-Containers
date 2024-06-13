# Deploy Qwen2 7B Instruct with Text Generation Inference in GKE from a GCS bucket

TL; DR TODO. Text Generation Inference (TGI) is a toolkit developed by Hugging Face for deploying and serving LLMs, with high performance text generation. And, Google Kubernetes Engine (GKE) is a fully-managed Kubernetes service in Google Cloud that can be used to deploy and operate containerized applications at scale using GCP's infrastructure. This post explains how to deploy an LLM from the Hugging Face Hub, as Llama3 8B Instruct, in a GKE Cluster running a purpose-built container to deploy LLMs in a secure and managed environment with the Hugging Face DLC for TGI.

## Setup / Configuration

First, we need to install both `gcloud` and `kubectl` in our local machine, which are the command-line tools for Google Cloud and Kubernetes, respectively, to interact with the GCP and the GKE Cluster.

* To install `gcloud`, follow the instructions at https://cloud.google.com/sdk/docs/install.
* To install `kubectl`, follow the instructions at https://kubernetes.io/docs/tasks/tools/#kubectl.

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

Once we are logged in, we need to enable the necessary services in GCP, such as the Google Kubernetes Engine API, the Google Container Registry API, and the Google Container File System API, which are necessary for the deployment of the GKE Cluster and the Hugging Face DLC for TGI.

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
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin.

And, if we're willing to upload a model from the Hugging Face Hub to a bucket in GCS, as in this example, then we will need to also install the `gsutil` tool, which is the command-line tool for Google Cloud Storage, to interact with the GCS.

```bash
gcloud components install gsutil
```

## Create GKE Cluster

Once we've set everything up, we are ready to start with the creation of the GKE Cluster and the node pool, which in this case will be a single GPU node, in order to use the GPU accelerator for high performance inference, also following TGI recommendations based on their internal optimizations for GPUs.

In order to deploy the GKE Cluster, we will use the "Autopilot" mode, which is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, one can also use the "Standard" mode.

> [!NOTE]
> Important to check before creating the GKE Autopilot Cluster https://cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus#before_you_begin, since not all the versions support GPU accelerators.

```bash
gcloud container clusters create-auto $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --release-channel=rapid \
    --cluster-version=1.30
```

> [!NOTE]
> To select the specific version in our location of the GKE Cluster, we can run the following command:
> ```bash
> gcloud container get-server-config \
>     --flatten="channels" \
>     --filter="channels.channel=RAPID" \
>     --format="yaml(channels.channel,channels.defaultVersion)" \
>     --location=$LOCATION
> ```
> For more information please visit https://cloud.google.com/kubernetes-engine/versioning#specifying_cluster_version.

If you prefer to use GKE Clusters from the stable channel, note that you may not have all the accelerator offering to your disposal, as the `nvidia-l4` which is only available from 1.28.3 onwards, not available yet in the stable channel.

As of the GKE documentation and service page in GCP, the creation of the GKE Cluster can take 5 minutes or more, depending on the configuration and the location of the cluster.

![GKE Cluster in the GCP Console](./imgs/gke-cluster.png)

## Optional: Upload a model from the Hugging Face Hub to GCS

This is an optional step in the tutorial, since you may want to re-use an existing model on a GCS bucket, if that's the case, then feel free to jump to the next step of the tutorial on how to configure the IAM for GCS so that you can access the bucket from a pod in the GKE Cluster.

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
> Make sure to set the proper permissions to run the script i.e. chmod +x ./scripts/upload_model_to_gcs.sh

```bash
./scripts/upload_model_to_gcs.sh --model-id Qwen/Qwen2-7B-Instruct --gcs gs://$BUCKET_NAME/Qwen2-7B-Instruct
```

![GCS Bucket in the GCP Console](./imgs/gcs-bucket.png)

## Configure IAM for GCS

Before we proceed with the deployment of the Hugging Face LLM DLC for TGI in the GKE Cluster, we need to set the IAM permissions for the GCS bucket so that the pod in the GKE Cluster can access the bucket. To do so, we will create a namespace and a service account in the GKE Cluster, and then set the IAM permissions for the GCS bucket that contains the model, either as uploaded from the Hugging Face Hub or as already existing in the GCS bucket.

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

## Deploy TGI

Once we are all set up, we can proceed to the Kubernetes deployment of the Hugging Face LLM DLC for TGI, serving the Qwen2 7B Instruct model from a volume mount under `/data` copied from the GCS bucket where the model is.

Then we can already deploy the Hugging Face LLM DLC for TGI via `kubectl`, from the following configuration files in the `configs/` directory:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face LLM DLC setting the `MODEL_ID` to the model path in the volume mount, in this case `/data/Qwen2-7B-Instruct`.
* `service.yaml`: contains the service details of the pod, exposing the port 80 for the TGI service.
* (optional) `ingress.yaml`: contains the ingress details of the pod, exposing the service to the external world so that it can be accessed via the ingress IP.

```bash
kubectl apply -f configs/
```

![GKE Deployment in the GCP Console](./imgs/gke-deployment.png)

> [!NOTE]
> The Kubernetes deployment may take a few minutes to be ready, so we can check the status of the deployment with the following command:
> ```bash
> kubectl get pods --namespace $NAMESPACE
> ```
> Alternatively, we can just wait for the deployment to be ready with the following command:
> ```bash
> kubectl wait --for=condition=Available --timeout=700s --namespace $NAMESPACE deployment/tgi-deployment
> ```

## Inference with TGI

In order to run the inference over the deployed TGI service, we can either:

* Port-forwarding the deployed TGI service to the port 8080, so as to access via `localhost` with the command:

    ```bash
    kubectl port-forward --namespace $NAMESPACE service/tgi-service 8080:8080
    ```

* Accessing the TGI service via the external IP of the ingress, which is the default scenario here since we have defined the ingress configuration in the `./configs/ingress.yaml` file (but it can be skipped in favour of the port-forwarding), that can be retrieved with the following command:

    ```bash
    kubectl get ingress --namespace $NAMESPACE tgi-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    ```

### Via cURL

To send a POST request to the TGI service using `cURL`, we can run the following command:

```bash
curl http://localhost:8080/generate \
    -X POST \
    -d '{"inputs":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n","parameters":{"temperature":0.7, "top_p": 0.95, "max_new_tokens": 128}}' \
    -H 'Content-Type: application/json'
```

Or to send the POST request to the ingress IP:

```bash
curl http://<ingress-ip>/generate \
    -X POST \
    -d '{"inputs":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n","parameters":{"temperature":0.7, "top_p": 0.95, "max_new_tokens": 128}}' \
    -H 'Content-Type: application/json'
```

Which produces the following output:

```bash
{"generated_text":"2 + 2 equals 4."}⏎
```

> [!NOTE]
> To generate the `inputs` with the expected chat template formatting, one could use the following snippet:
> ```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
> tokenizer.apply_chat_template(
>     [
>         {"role": "system", "content": "You are a helpful assistant."},
>         {"role": "user", "content": "What is 2+2?"},
>     ],
>     tokenize=False,
>     add_generation_prompt=True,
> )
> ```

### Via Python

To run the inference using Python, we can use the `openai` Python SDK (see the installation notes at https://platform.openai.com/docs/quickstart), setting the ingress IP as the `base_url` for the client, and then running the following code:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/",  # or http://<ingress-ip>/v1/
    api_key=os.getenv("HF_TOKEN"),
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    max_tokens=128,
)
```

Which produces the following output:

```bash
...
```

## Delete GKE Cluster

Finally, once we are done using TGI in the GKE Cluster, we can safely delete the cluster we've just created to avoid incurring in unnecessary costs.

```bash
gcloud container clusters delete $CLUSTER_NAME --location=$LOCATION
```

Alternatively, we can also downscale the replicas of the deployed pod to 0 in case we want to preserve the cluster, since the default GKE Cluster deployed with GKE Autopilot mode is running just a single `e2-small` instance.

```bash
kubectl scale --replicas=0 --namespace $NAMESPACE deployment/tgi-deployment
```
