# Deploy Mistral 7B on GKE 

This tutorial shows you how to serve a large language model (LLM) with GPUs in Google Kubernetes Engine (GKE) mode. This tutorial creates a GKE Standard cluster that L4 GPUs and prepares the GKE infrastructure. The tutorial use the `gcloud` command-line tool to create the cluster and deploy the model. Alternatively, you could use terraform to create the cluster and deploy the model. This example is not included in the terraform scripts.

We are going to deploy [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). 

## 1. Setup environment

We assume that you have a Google Cloud account and have installed the following packages: 
* [Google Cloud SDK](https://cloud.google.com/sdk/docs/install). 
* [GKE auth plugin](https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke?hl=en)
* [kubectl](https://kubernetes.io/docs/tasks/tools/)
* [yq](https://github.com/mikefarah/yq?tab=readme-ov-file#install)

First step is to login to your Google Cloud account, set the project id, the region, compute zone and enable the Google Kubernetes Engine API. 



```bash
export PROJECT_ID=
export REGION=
export CLUSTER_NAME=

gcloud auth login
gcloud config set project ${PROJECT_ID}
gcloud config set compute/region ${REGION}
# enable the Google Kubernetes Engine API
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable containerfilesystem.googleapis.com
```


## 2. Create a GKE Standard cluster and a GPU node pool

Create a Standard cluster that uses [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity):

```bash
MACHINE_TYPE=n2d-standard-4

gcloud container clusters create ${CLUSTER_NAME} \
  --workload-pool ${PROJECT_ID}.svc.id.goog \
  --enable-image-streaming \
  --node-locations=$REGION-a \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --addons GcsFuseCsiDriver   \
  --machine-type ${MACHINE_TYPE} \
  --num-nodes 1 --min-nodes 1 --max-nodes 1 \
  --ephemeral-storage-local-ssd=count=1 \
  --addons HttpLoadBalancing
```

Next we create a GPU node pool. We are creating the node pool scaled down to 0 nodes. You aren't paying for any GPUs until you start launching Kubernetes Pods that request GPUs. This node pool provisions [Spot VMs](https://cloud.google.com/kubernetes-engine/docs/how-to/spot-vms), which are priced lower than the default standard Compute Engine VMs. If you ran into availability issues, you can remove the `--spot` flag from this command, and the `cloud.google.com/gke-spot` node selector in the `mistral-tgi.yaml` config to use use on-demand VMs.

```bash
NODE_POOL_NAME=l4-gpu-pool
GPU=nvidia-l4
NUM_GPUS=1
GPU_MACHINE_TYPE=g2-standard-4

gcloud container node-pools create ${NODE_POOL_NAME} --cluster ${CLUSTER_NAME} \
  --accelerator type=${GPU},count=${NUM_GPUS},gpu-driver-version=latest \
  --machine-type ${GPU_MACHINE_TYPE} \
  --ephemeral-storage-local-ssd=count=1 \
  --enable-autoscaling --enable-image-streaming \
  --num-nodes=0 --min-nodes=0 --max-nodes=1 \
  --node-locations $REGION-a,$REGION-c --region $REGION --spot
```

Last step before we can deploy the model is to configure the kubectl context to use the new cluster:

``` 
gcloud container clusters get-credentials ${CLUSTER_NAME}
```

validate that the cluster is accessible and the GPU node pool is created:

```bash
kubectl get nodes -o wide
gcloud container node-pools list --cluster=$CLUSTER_NAME
```

## 3. Deploy Mistral 7B to GKE

We are going to deploy [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). We will deploy a single replica of the model with an Ingress controller. The Ingress controller is a Kubernetes resource that exposes the model to the internet. The Ingress controller is configured with a static IP address. 

Before we can deploy the model we need to update the MODEL_ID in the `mistral-tgi.yaml` file. 

```bash
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
yq e '.spec.template.spec.containers[0].env[] |= select(.name == "MODEL_ID").value = "'${MODEL_ID}'"' -i configs/deployment.yaml
```

Next we deploy the model:

```bash
kubectl apply -f configs/
```

We should see the following output:

```bash
deployment.apps/llm created
Warning: annotation "kubernetes.io/ingress.class" is deprecated, please use 'spec.ingressClassName' instead
ingress.networking.k8s.io/llm-ingress created
service/llm-service created
```

Ignore the warning, see https://cloud.google.com/kubernetes-engine/docs/concepts/ingress

We can check the status of the deployment with the following command:

```bash
kubectl get pods
```

_Note: It can take a few minutes for the model to download and start serving requests._

We can check the status of the Ingress controller with the following command:

```bash
kubectl get ingress llm-ingress 
```

_Note: It might take a few minutes for GKE to allocate an external IP address and set up forwarding rules before the load balancer is ready to serve your application. You might get errors such as HTTP 404 or HTTP 500 until the load balancer configuration is propagated across the globe._


## 4. Test the model

We can test the model with the following command:

```bash
INGRESS_IP=$(kubectl get ingress llm-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://${INGRESS_IP}/generate \
    -X POST \
    -d '{"inputs":"[INST] What is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```
You should see the following output:

```json
{"generated_text":" The sum of 10 and 10 is 20. So, 10 + 10 = 20."}
```


## 5. Clean up and delete the cluster

To avoid incurring charges to your Google Cloud account for the resources used in this tutorial, you can delete the GKE cluster and the GPU node pool.

```bash
gcloud container clusters delete ${CLUSTER_NAME}
```


## Optional Deploy Mistral on GKE Autopilot

## Resources used

https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-multiple-gpu
https://cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus
https://cloud.google.com/kubernetes-engine/docs/how-to/creating-an-autopilot-cluster
https://cloud.google.com/kubernetes-engine/docs/tutorials/http-balancer
