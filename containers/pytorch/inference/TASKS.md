## Hugging Face PyTorch DLC for Inference - Supported Tasks

Please find below all the supported tasks for each library at the time of writing this document:

### Transformers (WIP)

<details>
  <summary>text-classification</summary>
</details>

### Sentence Transformers

<details>
  <summary>sentence-similarity</summary>
  Sentence Similarity is the task of determining how similar two texts are. Sentence similarity models convert input texts into vectors (embeddings) that capture semantic information and calculate how close (similar) they are between them. This task is particularly useful for information retrieval and clustering/grouping.

It can be used via the [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) (running on top of the `SentenceTransformer` class from the [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) library) by setting the `HF_TASK` environment variable to `sentence-similarity` and the `HF_MODEL_ID` to the model ID of the model you want to deploy.

Below you can find an example with the environment variable values:

```bash
HF_MODEL_ID=BAAI/bge-m3
HF_TASK=sentence-similarity
```

More information about the sentence-similarity task at [Hugging Face Documentation - Sentence Similarity](https://huggingface.co/tasks/sentence-similarity) and at [Sentence Transformers Documentation - Sentence Transformer](https://sbert.net/docs/quickstart.html#sentence-transformer), and explore [all the supported sentence-similarity models on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=sentence-similarity&library=sentence-transformers&sort=trending).

</details>

<details>
  <summary>sentence-embeddings</summary>
  Sentence Embeddings is the task of converting input texts into vectors (embeddings) that capture semantic information. Sentence embeddings models are useful for a wide range of taskssuch as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.

It can be used via the [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) (running on top of the `SentenceTransformer` class from the [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) library) by setting the `HF_TASK` environment variable to `sentence-embeddings` and the `HF_MODEL_ID` to the model ID of the model you want to deploy.

Below you can find an example with the environment variable values:

```bash
HF_MODEL_ID=BAAI/bge-m3
HF_TASK=sentence-embeddings
```

More information about the sentence-embeddings task at [Sentence Transformers Documentation - Sentence Transformer](https://sbert.net/docs/quickstart.html#sentence-transformer), and explore [all the supported sentence-similarity models on the Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&sort=trending).

</details>

<details>
  <summary>sentence-ranking</summary>
  Sentence Ranking is the task of determining the relevance of a text to a query. Sentence ranking models convert input texts into vectors (embeddings) that capture semantic information and calculate how relevant they are to a query. This task is particularly useful for information retrieval and search engines.

It can be used via the [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) (running on top of the `CrossEncoder` class from the [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) library) by setting the `HF_TASK` environment variable to `sentence-ranking` and the `HF_MODEL_ID` to the model ID of the model you want to deploy.

Below you can find an example with the environment variable values:

```bash
HF_MODEL_ID=BAAI/bge-reranker-v2-m3
HF_TASK=sentence-ranking
```

More information about the sentence-ranking task at [Sentence Transformers Documentation - Cross Encoder](https://sbert.net/docs/quickstart.html#cross-encoder), and explore [all the supported sentence-ranking models on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-classification&library=sentence-transformers&sort=trending).

</details>

### Diffusers

<details>
  <summary>text-to-image</summary>
  Text-to-Image is a task that generates images from input text. These models can be used to generate and modify images based on text prompts.

It can be used via the [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) (running on top of the `AutoPipelineForText2Image` from the [`diffusers`](https://github.com/huggingface/diffusers) library) by setting the `HF_TASK` environment variable to `text-to-image` and the `HF_MODEL_ID` to the model ID of the model you want to deploy.

Below you can find an example with the environment variable values:

```bash
HF_MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TASK=text-to-image
```

More information about the text-to-image task at [Hugging Face Documentation - Text to Image](https://huggingface.co/tasks/text-to-image), and explore [all the supported text-to-image models on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-to-image&library=diffusers&sort=trending).

</details>

> [!NOTE]
> More tasks and models will be supported in the future, so please check [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) for the latest updates.
