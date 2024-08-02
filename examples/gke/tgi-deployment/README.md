# Deploy Llama3 8B with Text Generation Inference in GKE

TL; DR Llama 3 is the latest LLM from the Llama family, released by Meta; coming in two sizes 8B and 70B, including both the base model and the instruction-tuned model. Text Generation Inference (TGI) is a toolkit developed by Hugging Face for deploying and serving LLMs, with high performance text generation. And, Google Kubernetes Engine (GKE) is a fully-managed Kubernetes service in Google Cloud that can be used to deploy and operate containerized applications at scale using GCP's infrastructure. This post explains how to deploy an LLM from the Hugging Face Hub, as Llama3 8B Instruct, in a GKE Cluster running a purpose-built container to deploy LLMs in a secure and managed environment with the Hugging Face DLC for TGI.

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
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit <https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin>.

## Create GKE Cluster

Once we've set everything up, we are ready to start with the creation of the GKE Cluster and the node pool, which in this case will be a single GPU node, in order to use the GPU accelerator for high performance inference, also following TGI recommendations based on their internal optimizations for GPUs.

In order to deploy the GKE Cluster, we will use the "Autopilot" mode, which is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, one can also use the "Standard" mode.

> [!NOTE]
> Important to check before creating the GKE Autopilot Cluster <https://cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus#before_you_begin>, since not all the versions support GPU accelerators e.g. `nvidia-l4` is not supported in the GKE cluster versions 1.28.3 or lower.

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

## Optional: Set Secrets in GKE

Once the GKE Cluster is created then we can already proceed to the TGI deployment, but before that, we will create a Kubernetes secret for the GKE Cluster containing the Hugging Face Hub token, which may not be necessary in most of the cases, but it will be necessary for gated and private models, so we will showcase how to include it in case anyone wants to reproduce with a gated / private model.

In order to set the Kubernetes secret, we first need to get the credentials of the GKE Cluster so that we can access it via `kubectl`:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

Then we can already set the Kubernetes secret with the Hugging Face Hub token via `kubectl`. To generate a custom token for the Hugging Face Hub, you can follow the instructions at <https://huggingface.co/docs/hub/en/security-tokens>.

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_token=$HF_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -
```

![GKE Secret in the GCP Console](./imgs/gke-secrets.png)

More information on how to set Kubernetes secrets in a GKE Cluster at <https://cloud.google.com/secret-manager/docs/secret-manager-managed-csi-component>.

## Deploy TGI

Once we are all set up, we can proceed to the Kubernetes deployment of the Hugging Face LLM DLC for TGI, serving the Llama3 8B Instruct model from the Hugging Face Hub.

If not ran already within the previous step i.e. [Optional: Set Secrets in GKE](#optional-set-secrets-in-gke), we need to get the credentials of the GKE Cluster so that we can access it via `kubectl`:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

Then we can already deploy the Hugging Face LLM DLC for TGI via `kubectl`, from the following configuration files in the `config/` directory:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face LLM DLC setting the `MODEL_ID` to `meta-llama/Meta-Llama-3-8B-Instruct`.
* `service.yaml`: contains the service details of the pod, exposing the port 80 for the TGI service.
* (optional) `ingress.yaml`: contains the ingress details of the pod, exposing the service to the external world so that it can be accessed via the ingress IP.

```bash
kubectl apply -f config/
```

![GKE Deployment in the GCP Console](./imgs/gke-deployment.png)

> [!NOTE]
> The Kubernetes deployment may take a few minutes to be ready, so we can check the status of the deployment with the following command:
>
> ```bash
> kubectl get pods
> ```
>
> Alternatively, we can just wait for the deployment to be ready with the following command:
>
> ```bash
> kubectl wait --for=condition=Available --timeout=700s deployment/tgi-deployment
> ```

## Inference with TGI

In order to run the inference over the deployed TGI service, we can either:

* Port-forwarding the deployed TGI service to the port 8080, so as to access via `localhost` with the command:

    ```bash
    kubectl port-forward service/tgi-service 8080:8080
    ```

* Accessing the TGI service via the external IP of the ingress, which is the default scenario here since we have defined the ingress configuration in the `config/ingress.yaml` file (but it can be skipped in favour of the port-forwarding), that can be retrieved with the following command:

    ```bash
    kubectl get ingress tgi-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    ```

### Via cURL

To send a POST request to the TGI service using `cURL`, we can run the following command:

```bash
curl http://localhost:8080/generate \
    -X POST \
    -d '{"inputs":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","parameters":{"temperature":0.7, "top_p": 0.95, "max_new_tokens": 128}}' \
    -H 'Content-Type: application/json'
```

Or to send the POST request to the ingress IP:

```bash
curl http://<ingress-ip>/generate \
    -X POST \
    -d '{"inputs":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","parameters":{"temperature":0.7, "top_p": 0.95, "max_new_tokens": 128}}' \
    -H 'Content-Type: application/json'
```

Which produces the following output:

```bash
{"generated_text":"The answer to 2+2 is 4."}⏎
```

> [!NOTE]
> To generate the `inputs` with the expected chat template formatting, one could use the following snippet:
>
> ```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
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

To run the inference using Python, we can use the `openai` Python SDK (see the installation notes at <https://platform.openai.com/docs/quickstart>), setting the ingress IP as the `base_url` for the client, and then running the following code:

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
ChatCompletion(id='', choices=[Choice(finish_reason='eos_token', index=0, logprobs=None, message=ChatCompletionMessage(content='The answer to 2+2 is 4!', role='assistant', function_call=None, tool_calls=None))], created=1718108522, model='meta-llama/Meta-Llama-3-8B-Instruct', object='text_completion', system_fingerprint='2.0.2-sha-6073ece', usage=CompletionUsage(completion_tokens=12, prompt_tokens=28, total_tokens=40))
```

## Delete GKE Cluster

Finally, once we are done using TGI in the GKE Cluster, we can safely delete the cluster we've just created to avoid incurring in unnecessary costs.

```bash
gcloud container clusters delete $CLUSTER_NAME --location=$LOCATION
```

Alternatively, we can also downscale the replicas of the deployed pod to 0 in case we want to preserve the cluster, since the default GKE Cluster deployed with GKE Autopilot mode is running just a single `e2-small` instance.

```bash
kubectl scale --replicas=0 deployment/tgi-deployment
```
