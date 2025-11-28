# Scripts

This directory contains some useful scripts when working with Hugging Face in Google Cloud Platform (GCP), and some are used within some of the examples in order to provide the community with a simpler way to interact with Hugging Face via GCP and the other way around.

Note that before running any script, you will need to set the proper permissions to execute those, you could use the following command:

```bash
find . -type f -not -name "README.md" -exec chmod 755 {} \;
```

Additionally, it's mandatory to login into your Google Console, as well as setting the current active project as follows:

```bash
gcloud auth login
gcloud config set project <project_id>
```

Besides that, for the scripts that are accessing the Hugging Face Hub, is recommended to have `huggingface-cli` installed via `pip` as follows:

```bash
pip install "huggingface_hub[hf-xet]" --upgrade
```

And then, to either set the `HF_TOKEN` environment variable with a fine-grained token with the required permissions, or just to run `huggingface-cli login` and provide the same token there so that's included within the Hugging Face Hub cache.

## Available Scripts

* [`push_image_to_gcp.sh`](./push_image_to_gcp.sh): Script to push a locally built Docker image to Google Cloud's Artifact Registry, so that the Docker Artifact Registry will be created if not existing, including error handling and messages along the process.

* [`upload_model_to_gcs.sh`](./upload_model_to_gcs.sh): Script to download and upload a model from the Hugging Face Hub into a Google Cloud Storage (GCS) bucket, including the bucket creation if not existing, error handling and messages along the process.
