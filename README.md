# RAG-simple examples of exploration in BTBU
This example is a preliminary exploration of later work. It uses framework tools such as Ollama and Langchain to make a simple program that can be plugged into a knowledge base to perform RAG question and answer.
### What is RAG?
LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

🔷Retrieve:User questions are used to retrieve relevant context from external knowledge bases. To do this, the user query will be embedded in the same vector space as the "context in the vector database", and then a similarity search will be done in this space to return the top k data objects in the database that are most similar to the query.

🔷Augment:User queries and retrieved content are stuffed into a prompt template.

🔷Generate:Finally, the retrieval-enhanced cues are fed into the LLM.

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
## Quick Start
### Overall flow chart
<div align="left side">
 <img alt="Overall flow chart" height="px10" src="https://cdn-lfs-us-1.huggingface.co/repos/13/3d/133d8ca2460bf82ba2bdbe928d91a6c780364a6d0cf9005087db081cca492c02/ed22547b1538ea4fd18ea26777e14d9f7e51b3388b34d3cadf165cc37a7f63e0?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27RAG_workflow.png%3B+filename%3D%22RAG_workflow.png%22%3B&response-content-type=image%2Fpng&Expires=1720595234&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyMDU5NTIzNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzEzLzNkLzEzM2Q4Y2EyNDYwYmY4MmJhMmJkYmU5MjhkOTFhNmM3ODAzNjRhNmQwY2Y5MDA1MDg3ZGIwODFjY2E0OTJjMDIvZWQyMjU0N2IxNTM4ZWE0ZmQxOGVhMjY3NzdlMTRkOWY3ZTUxYjMzODhiMzRkM2NhZGYxNjVjYzM3YTdmNjNlMD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=LV%7E8WlQ2VBRcTeKDaFyH-ZiwsnQ76OKtGqa8vRzftKIY1qqY8vRNI7QP0H9ApdCXdBOVdMnmHMOJlq2yKnYPoQk7O5Q5O1dDcUPhbyuEvnuIIuoVNThmRxq-PCjbCuaaUuuU8jpMWW-K%7EFp2kssB2-KXX5NmLJ-qV7Xdl9jCdDCKr2JbyJN5Wach5p5rRlfh-sdB2a3vESqFI3pHvGgM7XCDbh1JmD0925Q7pvrRMXfFoBy-Ibvu31xiKnu%7EV%7EnvtVVtmtYTNrTSCI93xZ0DWhpSCOzoCP51wobl9fyoM5N9It4HXQeoxUoM8aA0wmirJLy7OtJC2yHuWnCNO3FLAg__&Key-Pair-Id=K24J24Z295AEI9">
</div>

<div align="right side">
 <img alt="overall flow chart" height="px10" src="https://s2.loli.net/2024/07/07/OUPa9gmAxZ15elW.png">
</div>
This seems like a extremely complicated process,right?
Let's look at it separately.

### 1.Flow Chart(pre-production)
<div align="center">
 <img alt="flow chart" height="px150" src="https://python.langchain.com/v0.2/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png">
</div>

#### 1.Load
Langchain provides many built-in document loaders<a href="https://python.langchain.com/v0.2/docs/how_to/#document-loaders/" target="_blank">(Documents-loaders)</a>.

A document is a dictionary containing text and raw data. Here, we use PyMuPDF to process the content in PDF files better.
```bash
import fitz  # PyMuPDF

def read_pdf(file_path):
    doc = fitz.open(file_path)
    content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        content += page.get_text()
    return content

def pdf_to_documents(file_path):
    content = read_pdf(file_path)
    return [Document(page_content=content)]

file_path = "your storage location/liulang.pdf"
raw_documents = pdf_to_documents(file_path)
```
#### 2.Split/Chunk
Document Chunking:Text splitter will split files or text into chunks to prevent the file information from exceeding LLM tokens. The tools commonly used in this step are RecursiveCharacterTextSplitter and CharacterTextSplitter. The difference is that RecursiveCharacterTextSplitter will also recursively split the text into smaller blocks if the block size exceeds the specified threshold.

🔸chunk_size: Determines the maximum number of characters in each chunk when splitting text. It specifies the size or length of each block.

🔸chunk_overlap: Determines the number of characters that overlap between consecutive blocks when splitting text. It specifies how much of the previous block should be included in the next block.

Because the original document is too long to be directly input into our large model, the document needs to be cut into small pieces first. Langchain also provides many built-in text segmentation tools. Here we use RecursiveCharacterTextSplitter, set chunk_size to 500, and chunk_overlap to 50 to ensure text continuity between chunks.

Langchain also provides a variety of text-splitter for you to choose.<a href="https://python.langchain.com/v0.2/docs/how_to/#text-splitters">(Text Splitters)</a>.

🟢BTW:This is a very useful little tool that can clearly see text chunking case.<a href="https://chunkviz.up.railway.app/">(Chunkviz)</a>.
```bash
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(raw_documents)
```
#### 3.Embed
Embedding:To enable semantic search over text blocks, we need to generate vector embeddings for each block and then store them together with their embeddings. Ollama’s embedding model used here.
