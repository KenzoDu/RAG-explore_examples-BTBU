# RAG-simple examples of exploration in BTBU
This example is a preliminary exploration of later work. It uses framework tools such as Ollama and Langchain to make a simple program that can be plugged into a knowledge base to perform RAG question and answer.
## What is RAG?
LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).Welcome to Verba: The Golden RAGtriever, an open-source application designed to offer an end-to-end, streamlined, and user-friendly interface for Retrieval-Augmented Generation (RAG) out of the box. In just a few easy steps, explore your datasets and extract insights with ease, either locally with HuggingFace and Ollama or through LLM providers such as OpenAI, Cohere, and Google.

## Preliminary work

<div align="left side">
 <img alt="ollama" height="100px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

### Ollama
Get up and running with large language models.


#### Windows preview

[Download](https://ollama.com/download/OllamaSetup.exe)

#### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

#### Linux

```
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md)```
pip install goldenverba
```

![Demo of Verba](https://github.com/weaviate/Verba/blob/1.0.0/img/verba.gif)

- [Verba](#verba)
  - [🎯 What Is Verba?](#what-is-verba)
  - [✨ Features](#feature-lists)
- [✨ Getting Started with Verba](#getting-started-with-verba)
- [🔑 API Keys](#api-keys)
  - [Weaviate](#weaviate)
  - [Ollama](#ollama)
  - [Google](#google)
  - [Unstructured](#unstructured)
  - [OpenAI](#openai)
  - [HuggingFace](#huggingface)
- [Quickstart: Deploy with pip](#how-to-deploy-with-pip)
- [Quickstart: Build from Source](#how-to-build-from-source)
- [Quickstart: Deploy with Docker](#how-to-install-verba-with-docker)
- [💾 Verba Walkthrough](#️verba-walkthrough)
- [💖 Open Source Contribution](#open-source-contribution)
- [🚩 Known Issues](#known-issues)
- [❔FAQ](#faq)

## What Is Verba?

Verba is a fully-customizable personal assistant for querying and interacting with your data, **either locally or deployed via cloud**. Resolve questions around your documents, cross-reference multiple data points or gain insights from existing knowledge bases. Verba combines state-of-the-art RAG techniques with Weaviate's context-aware database. Choose between different RAG frameworks, data types, chunking & retrieving techniques, and LLM providers based on your individual use-case.

## Open Source Spirit

**Weaviate** is proud to offer this open-source project for the community. While we strive to address issues promptly, please understand that it may not be maintained with the same rigor as production software. We welcome and encourage community contributions to help keep it running smoothly. Your support in fixing open issues quickly is greatly appreciated.

### Watch our newest Verba video here:

[![VIDEO LINK](https://github.com/weaviate/Verba/blob/main/img/thumbnail.png)](https://www.youtube.com/watch?v=swKKRdLBhas&t)

## Feature Lists

| 🤖 Model Support                  | Implemented | Description                                             |
| --------------------------------- | ----------- | ------------------------------------------------------- |
| Ollama (e.g. Llama3)              | ✅          | Local Embedding and Generation Models powered by Ollama |
| HuggingFace (e.g. MiniLMEmbedder) | ✅          | Local Embedding Models powered by HuggingFace           |
| Cohere (e.g. Command R+)          | ✅          | Embedding and Generation Models by Cohere               |
| Google (e.g. Gemini)              | ✅          | Embedding and Generation Models by Google               |
| OpenAI (e.g. GPT4)                | ✅          | Embedding and Generation Models by OpenAI               |

| 🤖 Embedding Support | Implemented | Description                              |
| -------------------- | ----------- | ---------------------------------------- |
| Ollama               | ✅          | Local Embedding Models powered by Ollama |
| MiniLMEmbedder       | ✅          | powered by HuggingFace                   |
| AllMPNetEmbedder     | ✅          | powered by HuggingFace                   |
| MixedbreadEmbedder   | ✅          | powered by HuggingFace                   |
| Cohere               | ✅          | Embedding Models by Cohere               |
| Google               | ✅          | Embedding Models by Google               |
| OpenAI               | ✅          | Embedding Models by OpenAI               |

| 📁 Data Support    | Implemented | Description                         |
| ------------------ | ----------- | ----------------------------------- |
| PDF Ingestion      | ✅          | Import PDF into Verba               |
| GitHub & GitLab    | ✅          | Import Files from Github and GitLab |
| CSV/XLSX Ingestion | ✅          | Import Table Data into Verba        |
| .DOCX              | ✅          | Import .docx files                  |
| Multi-Modal        | planned ⏱️  | Import Multi-Modal Data into Verba  |
| UnstructuredIO     | ✅          | Import Data through Unstructured    |

| ✨ RAG Features         | Implemented | Description                                                               |
| ----------------------- | ----------- | ------------------------------------------------------------------------- |
| Hybrid Search           | ✅          | Semantic Search combined with Keyword Search                              |
| Semantic Caching        | ✅          | Results saved and retrieved based on semantic meaning                     |
| Autocomplete Suggestion | ✅          | Verba suggests autocompletion                                             |
| Filtering               | planned ⏱️  | Apply Filters (e.g. documents, document types etc.) before performing RAG |
| Advanced Querying       | planned ⏱️  | Task Delegation Based on LLM Evaluation                                   |
| Reranking               | planned ⏱️  | Rerank results based on context for improved results                      |
| RAG Evaluation          | planned ⏱️  | Interface for Evaluating RAG pipelines                                    |
| Customizable Metadata   | planned ⏱️  | Free control over Metadata                                                |

| 🆒 Cool Bonus         | Implemented | Description                                             |
| --------------------- | ----------- | ------------------------------------------------------- |
| Docker Support        | ✅          | Verba is deployable via Docker                          |
| Customizable Frontend | ✅          | Verba's frontend is fully-customizable via the frontend |

| 🤝 RAG Libraries | Implemented | Description                        |
| ---------------- | ----------- | ---------------------------------- |
| Haystack         | planned ⏱️  | Implement Haystack RAG pipelines   |
| LlamaIndex       | planned ⏱️  | Implement LlamaIndex RAG pipelines |
| LangChain        | planned ⏱️  | Implement LangChain RAG pipelines  |

> Something is missing? Feel free to create a new issue or discussion with your idea!

![Showcase of Verba](https://github.com/weaviate/Verba/blob/1.0.0/img/verba_screen.png)

---

# Getting Started with Verba

You have three deployment options for Verba:

- Install via pip

```
pip install goldenverba
```

- Build from Source

```
git clone https://github.com/weaviate/Verba

pip install -e .
```

- Use Docker for Deployment

**Prerequisites**: If you're not using Docker, ensure that you have `Python >=3.10.0` installed on your system.

If you're unfamiliar with Python and Virtual Environments, please read the [python tutorial guidelines](./PYTHON_TUTORIAL.md).

# API Keys

Before starting Verba you'll need to configure access to various components depending on your chosen technologies, such as OpenAI, Cohere, and HuggingFace via an `.env` file. Create this `.env` in the same directory you want to start Verba in. You can find an `.env.example` file in the [goldenverba](./goldenverba/.env.example) directory.

> Make sure to only set environment variables you intend to use, environment variables with missing or incorrect values may lead to errors.

Below is a comprehensive list of the API keys and variables you may require:

| Environment Variable           | Value                                                      | Description                                                                                                            |
| ------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| WEAVIATE_URL_VERBA             | URL to your hosted Weaviate Cluster                        | Connect to your [WCS](https://console.weaviate.cloud/) Cluster                                                         |
| WEAVIATE_API_KEY_VERBA         | API Credentials to your hosted Weaviate Cluster            | Connect to your [WCS](https://console.weaviate.cloud/) Cluster                                                         |
| OPENAI_API_KEY                 | Your API Key                                               | Get Access to [OpenAI](https://openai.com/) Models                                                                     |
| OPENAI_BASE_URL                | URL to OpenAI instance                                     | Models                                                                                                                 |
| COHERE_API_KEY                 | Your API Key                                               | Get Access to [Cohere](https://cohere.com/) Models                                                                     |
| OLLAMA_URL                     | URL to your Ollama instance (e.g. http://localhost:11434 ) | Get Access to [Ollama](https://ollama.com/) Models                                                                     |
| OLLAMA_MODEL                   | Model Name (e.g. llama)                                    | Get Access to a specific [Ollama](https://ollama.com/) Model                                                           |
| OLLAMA_EMBED_MODEL             | Model Name (e.g. mxbai-embed-large)                        | Get Access to a specific [Ollama](https://ollama.com/) Model for embedding (Defaults to OLLAMA_MODEL if not specified) |
| UNSTRUCTURED_API_KEY           | Your API Key                                               | Get Access to [Unstructured](https://docs.unstructured.io/welcome) Data Ingestion                                      |
| UNSTRUCTURED_API_URL           | URL to Unstructured Instance                               | Get Access to [Unstructured](https://docs.unstructured.io/welcome) Data Ingestion                                      |
| GITHUB_TOKEN                   | Your GitHub Token                                          | Get Access to Data Ingestion via GitHub                                                                                |
| GITLAB_TOKEN                   | Your GitLab Token                                          | Get Access to Data Ingestion via GitLab                                                                                |
| GOOGLE_APPLICATION_CREDENTIALS | Google Credentials                                         | Get Access to Google Models                                                                                            |
| GOOGLE_CLOUD_PROJECT           | Google Cloud Project                                       | Get Access to Google Models                                                                                            |
| GOOGLE_API_KEY                 | Your API Key                                               | Get Access to Google Models                                                                                            |
| VERBA_PRODUCTION               | True                                                       | Run Verba in Production Mode                                                                                           |

## Weaviate

Verba provides flexibility in connecting to Weaviate instances based on your needs. By default, Verba opts for [Weaviate Embedded](https://weaviate.io/developers/weaviate/installation/embedded) if it doesn't detect the `WEAVIATE_URL_VERBA` and `WEAVIATE_API_KEY_VERBA` environment variables. This local deployment is the most straightforward way to launch your Weaviate database for prototyping and testing.

However, you also have other options:

**🌩️ Weaviate Cloud Service (WCS)**

If you prefer a cloud-based solution, Weaviate Cloud Service (WCS) offers a scalable, managed environment. Learn how to set up a cloud cluster and get the API keys by following the [Weaviate Cluster Setup Guide](https://weaviate.io/developers/wcs/guides/create-instance).

**🐳 Docker Deployment**
Another robust local alternative is deploying Weaviate using Docker. For more details, consult the [Weaviate Docker Guide](https://weaviate.io/developers/weaviate/installation/docker-compose).

## Ollama

Verba supports Ollama models. Download and Install Ollama on your device (https://ollama.com/download). Make sure to install your preferred LLM using `ollama run <model>`.

Tested with `llama3`, `llama3:70b` and `mistral`. The bigger models generally perform better, but need more computational power.

> Ensure that you have the right configurations for the `Embedder` and `Generator` selected before going ahead.

![verba-embedder](https://github.com/weaviate/Verba/blob/main/img/verba_select_embedder.png)

> Make sure Ollama Server runs in the background and that you don't ingest documents with different ollama models since their vector dimension can vary that will lead to errors

You can verify that by running the following command

```
ollama run llama3
```

![verba-ollama-llama3](https://github.com/weaviate/Verba/blob/main/img/ollama_running.png)

## Google

If you want to use the Google Features, make sure to install the Google Verba package.

```bash
pip install goldenverba[google]

or

pip install `.[google]`
```

> If you're using Docker, modify the Dockerfile accordingly

### Google Embeddings

For the Google Embeddings, Verba is using Vertex AI Studio inside Google Cloud. You can find instructions for obtaining a key [here](https://cloud.google.com/iam/docs/create-short-lived-credentials-direct). If you have the `gcloud` CLI installed, you can run the following command: `gcloud auth print-access-token`. **At the moment, this access token must be renewed every hour.**

You also need to set the `GOOGLE_CLOUD_PROJECT` environment variable to the name of your project.

### Google Gemini

To use Google Gemini, you need a service account key, which is a JSON file. To obtain this, go to "project settings" in your Google Cloud console, then to "service accounts". Create a new service account, then create a new key. Download this key and place it in the route of Verba. Name it `gemini_secrets.json` to have it excluded from git automatically. Set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the location of this file, e.g. `gemini_secrets.json`.

You also need to set the `GOOGLE_CLOUD_PROJECT` environment variable to the name of your project.

## Unstructured

Verba supports importing documents through Unstructured IO (e.g plain text, .pdf, .csv, and more). To use them you need the `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_API_URL` environment variable. You can get it from [Unstructured](https://unstructured.io/)

> UNSTRUCTURED_API_URL is set to `https://api.unstructured.io/general/v0/general` by default

## OpenAI

Verba supports OpenAI Models such as Ada, GPT3, and GPT4. To use them, you need to specify the `OPENAI_API_KEY` environment variable. You can get it from [OpenAI](https://openai.com/)

You can also add a `OPENAI_BASE_URL` to use proxies such as LiteLLM (https://github.com/BerriAI/litellm)

```
OPENAI_BASE_URL=YOUR-OPENAI_BASE_URL
```

### Azure OpenAI

To use Azure OpenAI, you need to set

- The API type:

```
OPENAI_API_TYPE="azure"
```

- The key and the endpoint:

```
OPENAI_API_KEY=<YOUR_KEY>
OPENAI_BASE_URL=http://XXX.openai.azure.com
```

- Azure OpenAI resource name, which is XXX if your endpoint is XXX.openai.azure.com

```
AZURE_OPENAI_RESOURCE_NAME=<YOUR_AZURE_RESOURCE_NAME>
```

- You need to set the models, for the embeddings and for the query.

```
AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"
OPENAI_MODEL="gpt-4"
```

- Finally, as Azure is using per-minute quota, you might need to add a waiting time between each chunk upload. For example, if you have a limit of 240k tokens per minute, if your chunks are
  400 tokens max, then 100ms between queries should be fine. If you get error 429 from weaviate, then increase this value.

```
WAIT_TIME_BETWEEN_INGESTION_QUERIES_MS="100"
```

## HuggingFace

If you want to use the HuggingFace Features, make sure to install the correct Verba package. It will install required packages to use the local embedding models.
Please note that on startup, Verba will automatically download and install all embedding models, if you just want specific models, please remove unwanted models from the `goldenverba/compoonents/managers.py` file.

```bash
pip install goldenverba[huggingface]

or

pip install `.[huggingface]`
```

> If you're using Docker, modify the Dockerfile accordingly

# How to deploy with pip

`Python >=3.10.0`

1. **Initialize a new Python Environment**

```
python3 -m virtualenv venv
```

2. **Install Verba**

```
pip install goldenverba
```

3. **Launch Verba**

```
verba start
```

> You can specify the --port and --host via flags

4. **Access Verba**

```
Visit localhost:8000
```

5. **Create .env file and add environment variables**

# How to build from Source

1. **Clone the Verba repos**

```
git clone https://github.com/weaviate/Verba.git
```

2. **Initialize a new Python Environment**

```
python3 -m virtualenv venv
```

3. **Install Verba**

```
pip install -e .
```

4. **Launch Verba**

```
verba start
```

> You can specify the --port and --host via flags

5. **Access Verba**

```
Visit localhost:8000
```

6. **Create .env file and add environment variables**

# How to install Verba with Docker

Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. To get started with deploying Verba using Docker, follow the steps below. If you need more detailed instructions on Docker usage, check out the [Docker Curriculum](https://docker-curriculum.com/).

0. **Clone the Verba repos**
   Ensure you have Git installed on your system. Then, open a terminal or command prompt and run the following command to clone the Verba repository:

```
git clone https://github.com/weaviate/Verba.git
```

1. **Set neccessary environment variables**
   Make sure to set your required environment variables in the `.env` file. You can read more about how to set them up in the [API Keys Section](#api-keys)

2. **Adjust the docker-compose file**
   You can use the `docker-compose.yml` to add required environment variables under the `verba` service and can also adjust the Weaviate Docker settings to enable Authentification or change other settings of your database instance. You can read more about the Weaviate configuration in our [docker-compose documentation](https://weaviate.io/developers/weaviate/installation/docker-compose)

> Please make sure to only add environment variables that you really need. If you have no authentifcation enabled in your Weaviate Cluster, make sure to not include the `WEAVIATE_API_KEY_VERBA` enviroment variable

2. **Deploy using Docker**
   With Docker installed and the Verba repository cloned, navigate to the directory containing the Docker Compose file in your terminal or command prompt. Run the following command to start the Verba application in detached mode, which allows it to run in the background:

```bash

docker compose up -d

```

```bash

docker compose --env-file .env up -d

```

This command will download the necessary Docker images, create containers, and start Verba.
Remember, Docker must be installed on your system to use this method. For installation instructions and more details about Docker, visit the official Docker documentation.

4. **Access Verba**

- You can access your local Weaviate instance at `localhost:8080`

- You can access the Verba frontend at `localhost:8000`

If you want your Docker Instance to install a specific version of Verba you can edit the `Dockerfile` and change the installation line.

```
RUN pip install -e '.'
```
