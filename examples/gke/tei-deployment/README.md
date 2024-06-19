# Deploy Snowflake's Artic Embed (M) with Text Embeddings Inference (TEI) in GKE

TL; DR Snowflake's Artic Embed is a suite of text embedding models that focuses on creating high-quality retrieval models optimized for performance, achieving state-of-the-art (SOTA) performance on the MTEB/BEIR leaderboard for each of their size variants. Text Embeddings Inference (TEI) is a toolkit developed by Hugging Face for deploying and serving open source text embeddings and sequence classification models; enabling high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5. And, Google Kubernetes Engine (GKE) is a fully-managed Kubernetes service in Google Cloud that can be used to deploy and operate containerized applications at scale using GCP's infrastructure. This post explains how to deploy a text embedding model from the Hugging Face Hub in a GKE Cluster running a purpose-built container to deploy text embedding models in a secure and managed environment with the Hugging Face DLC for TEI.

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
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin.

## Create GKE Cluster

Once we've set everything up, we are ready to start with the creation of the GKE Cluster and the node pool, which in this case will be a single CPU node as for most of the workloads CPU inference is enough to serve most of the text embeddings models, while it could benefit a lot from GPU serving.

> [!NOTE]
> CPU is being used to run the inference on top of the text embeddings models to showcase the current capabilities of TEI, but switching to GPU is as easy as replacing `spec.containers[0].image` with `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-2.ubuntu2204`, and then updating the requested resources as well as the `nodeSelector` requirements in the `deployment.yaml` file. For more information, please refer to the [`gpu-config`](./gpu-config/) directory that contains a pre-defined configuration for GPU serving in TEI with an NVIDIA Tesla T4 GPU (with a compute capability of 7.5 i.e. natively supported in TEI).

In order to deploy the GKE Cluster, we will use the "Autopilot" mode, which is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, one can also use the "Standard" mode.

> [!NOTE]
> Important to check before creating the GKE Autopilot Cluster https://cloud.google.com/kubernetes-engine/docs/how-to/performance-pods, since not all the versions support CPUs, and the CPUs vary depending on the workload.

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

As of the GKE documentation and service page in GCP, the creation of the GKE Cluster can take 5 minutes or more, depending on the configuration and the location of the cluster.

![GKE Cluster in the GCP Console](./imgs/gke-cluster.png)

## Optional: Set Secrets in GKE

Once the GKE Cluster is created then we can already proceed to the TEI deployment, but before that, we will create a Kubernetes secret for the GKE Cluster containing the Hugging Face Hub token, which may not be necessary in most of the cases, but it will be necessary for gated and private models, so we will showcase how to include it in case anyone wants to reproduce with a gated / private model.

In order to set the Kubernetes secret, we first need to get the credentials of the GKE Cluster so that we can access it via `kubectl`:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

Then we can already set the Kubernetes secret with the Hugging Face Hub token via `kubectl`. To generate a custom token for the Hugging Face Hub, you can follow the instructions at https://huggingface.co/docs/hub/en/security-tokens.

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_token=$HF_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -
```

![GKE Secret in the GCP Console](./imgs/gke-secrets.png)

More information on how to set Kubernetes secrets in a GKE Cluster at https://cloud.google.com/secret-manager/docs/secret-manager-managed-csi-component.

## Deploy TEI

Once we are all set up, we can proceed to the Kubernetes deployment of the Hugging Face LLM DLC for TEI, serving the [`Snowflake/snowflake-arctic-embed-m`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) model from the Hugging Face Hub.

Recently, the Hugging Face Hub team has included the `text-embeddings-inference` tag in the Hub, so feel free to explore all the embedding models in the Hub that can be served via TEI at https://huggingface.co/models?other=text-embeddings-inference.

If not ran already within the previous step i.e. [Optional: Set Secrets in GKE](#optional-set-secrets-in-gke), we need to get the credentials of the GKE Cluster so that we can access it via `kubectl`:

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

Then we can already deploy the Hugging Face LLM DLC for TEI via `kubectl`, from the following configuration files in the `configs/` directory:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face LLM DLC setting the `MODEL_ID` to [`Snowflake/snowflake-arctic-embed-m`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m).
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
> ```bash
> kubectl get pods
> ```
> Alternatively, we can just wait for the deployment to be ready with the following command:
> ```bash
> kubectl wait --for=condition=Available --timeout=700s deployment/tei-deployment
> ```

## Inference with TEI

In order to run the inference over the deployed TEI service, we can either:

* Port-forwarding the deployed TEI service to the port 8080, so as to access via `localhost` with the command:

    ```bash
    kubectl port-forward service/tei-service 8080:8080
    ```

* Accessing the TEI service via the external IP of the ingress, which is the default scenario here since we have defined the ingress configuration in the `./configs/ingress.yaml` file (but it can be skipped in favour of the port-forwarding), that can be retrieved with the following command:

    ```bash
    kubectl get ingress tei-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    ```

> [!NOTE]
> TEI exposes different inference endpoints based on the task that the model is serving:
>   - Text Embeddings: text embedding models expose the endpoint `/embed` expecting a payload with the key `inputs` which is either a string or a list of strings to be embedded.
>   - Re-rank: re-ranker models expose the endpoint `/rerank` expecting a payload with the keys `query` and `texts`, where the `query` is the reference used to rank the similarity against each text in `texts`.
>   - Sequence Classification: classic sequence classification models expose the endpoint `/predict` which expects a payload with the key `inputs` which is either a string or a list of strings to classify.
> More information at https://huggingface.co/docs/text-embeddings-inference/quick_tour.

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

Which produces the following output (truncated for brevity, but original tensor length is 768, which is the embedding dimension of [`Snowflake/snowflake-arctic-embed-m`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) i.e. the model we're serving):

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
