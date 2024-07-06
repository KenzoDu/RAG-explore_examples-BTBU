# RAG-simple examples of exploration in BTBU
This example is a preliminary exploration of later work. It uses framework tools such as Ollama and Langchain to make a simple program that can be plugged into a knowledge base to perform RAG question and answer.
### What is RAG?
LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

## Preliminary work

<div align="left side">
 <img alt="ollama" height="100px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

### Ollama*
Get up and running with large language models.


#### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

#### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

#### Model
This example uses the qwen model(just 2.3G)
```bash
pip install qwen:4b
```
### 🦜️🔗 LangChain*

#### Quick Install

With pip:
```bash
pip install langchain
```
With conda:
```bash
conda install langchain -c conda-forge
```
### PyMuPDF*
**PyMuPDF** requires **Python 3.8 or later**, install using **pip** with:
#### Installation
```bash
pip install PyMuPDF
```
### Chroma*

<p align="left side">
    <b>Chroma - the open-source embedding database</b>. <br />
    The fastest way to build Python or JavaScript LLM apps with memory!
</p>

  <a href="https://docs.trychroma.com/" target="_blank">
      Docs
  </a> |
  <a href="https://www.trychroma.com/" target="_blank">
      Homepage
  </a>
</p>


```bash
pip install chromadb # python client
# for javascript, npm install chromadb!
# for client-server mode, chroma run --path /chroma_db_path
```
