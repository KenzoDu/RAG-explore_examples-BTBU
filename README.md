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
```
pip install qwen:4b
```
### 🦜️🔗 LangChain*

#### Quick Install

With pip:
```
pip install langchain
```
With conda:
```
conda install langchain -c conda-forge
```
### PyMuPDF*
**PyMuPDF** requires **Python 3.8 or later**, install using **pip** with:
#### Installation
```
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


```
pip install chromadb # python client
# for javascript, npm install chromadb!
# for client-server mode, chroma run --path /chroma_db_path
```
## Documentation Guidelines

📌 **Whole Flow** :Complete full-process RAG system

📌 **Two-Step Flow**:Implement the RAG system in two steps, which can be more convenient to implement

📌 **liulangdiqiu.pdf**:流浪地球 Sample PDF

📌 **readme**:Code guidelines


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

#### a.Load
Langchain provides many built-in document loaders<a href="https://python.langchain.com/v0.2/docs/how_to/#document-loaders/" target="_blank">(Documents-loaders)</a>.

A document is a dictionary containing text and raw data. Here, we use PyMuPDF to process the content in PDF files better.
```
import fitz  # PyMuPDF
from langchain.schema import Document

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
#### b.Split/Chunk
Document Chunking:Text splitter will split files or text into chunks to prevent the file information from exceeding LLM tokens. The tools commonly used in this step are RecursiveCharacterTextSplitter and CharacterTextSplitter. The difference is that RecursiveCharacterTextSplitter will also recursively split the text into smaller blocks if the block size exceeds the specified threshold.

🔸chunk_size: Determines the maximum number of characters in each chunk when splitting text. It specifies the size or length of each block.

🔸chunk_overlap: Determines the number of characters that overlap between consecutive blocks when splitting text. It specifies how much of the previous block should be included in the next block.

Because the original document is too long to be directly input into our large model, the document needs to be cut into small pieces first. Langchain also provides many built-in text segmentation tools. Here we use RecursiveCharacterTextSplitter, set chunk_size to 500, and chunk_overlap to 50 to ensure text continuity between chunks.

Langchain also provides a variety of text-splitter for you to choose.<a href="https://python.langchain.com/v0.2/docs/how_to/#text-splitters">(Text Splitters)</a>.

🟢BTW:This is a very useful little tool that can clearly see text chunking case.<a href="https://chunkviz.up.railway.app/">(Chunkviz)</a>.
```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(raw_documents)
```
#### c.Embed
Embedding:Then use Embedding to convert the chunk text divided in previous step into vectors. LangChain provides many Embedding model interfaces, such as OpenAI, Cohere, Hugging Face, Weaviate, etc. You can refer to the LangChain official website.<a href="https://python.langchain.com/v0.2/docs/how_to/#embedding-models">(Embedding)</a>.

Ollama’s embedding model used here.

```bash
from langchain_community.embeddings import OllamaEmbeddings

embedding_model = OllamaEmbeddings()
```
#### d.Vector Stores

Store:We will store the results after Embedding in VectorDB. Common VectorDBs include Chroma, Pinecone, FAISS, etc. Here I use Chroma to implement it. Chroma is well integrated with LangChain, and you can directly use LangChain's interface.<a href="https://python.langchain.com/v0.2/docs/how_to/#vector-stores">(Vector-stores)</a>.

Extracts the text content of each document from the list all_splits containing multiple Document objects and stores the content in the list texts .
```
texts = [doc.page_content for doc in all_splits]
```
Persistence storage path of vector database
```
persist_directory = 'your storage location/chroma_db'
```
This code uses the Chroma class to convert the text list texts into vectors, stores these vectors in a collection named "RAG_chroma", the storage path is 'your storage location/chroma_db', and calls the persist method to persist the data
```
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="RAG_chroma",
    persist_directory=persist_directory
)
vectorstore.persist() 
```
### 2.Flow chart(In-production)

<div align="center">
 <img alt="overall flow chart" height="px10" src="https://python.langchain.com/v0.2/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png">
</div>

#### z.Load(Optional)

If you want to call the database at any time to implement Q&A, you must load the contents of the database first.

```
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

chroma_db_path = "your storage location\chroma_db"
embedding_model = OllamaEmbeddings()
vectorstore = Chroma.from_texts(
    embedding=embedding_model,
    texts=chroma_db_path,
    collection_name="RAG_chroma"
)
```
#### 📌Tips- Sorting process

Above we imported the PDF information into the DB and started the LLM service. Next we need to string together the entire RAG steps:

1️⃣ Users sent QA

2️⃣ Text Retrieval from Chroma_DB

3️⃣ Combine QA with Text Retrieval and send to LLM

4️⃣ LLM answers based on information

#### a.Retriever

First we need to create a Retriever, which can return corresponding files based on unstructured QA. LangChain provides many methods and integrates third-party tools. I use the Vectorstore method here. For other types, you can refer to <a href="https://python.langchain.com/v0.2/docs/how_to/#retrievers/">(Retriever)</a>.

```
retriever = vectorstore.as_retriever()
```

#### b.Prompt templates

Prompt templates help to translate user input and parameters into instructions for a language model. This can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based output.Prompt Templates take as input a dictionary, where each key represents a variable in the prompt template to fill in.Prompt Templates output a PromptValue. This PromptValue can be passed to an LLM or a ChatModel, and can also be cast to a string or a list of messages. The reason this PromptValue exists is to make it easy to switch between strings and messages.

There are a few different types of prompt templates.<a href="https://python.langchain.com/v0.2/docs/concepts/#prompt-templates">(prompt-templates)</a>.
This example uses ChatPromptTemplate.

##### b1.Template
Template defines a structure that formats the input data (context and question) and generates a complete text template, which is then passed to a large language model (LLM) to generate the final answer.

• {context}: This placeholder will be replaced by the relevant context retrieved from the vector store.

• {question}: This placeholder will be replaced by the entered question from users.
```
template = """Answer the question with Chinese and based only on the following context:
{context}

Question: {question}
"""
```
📍According to the PDF content in our example

•context:“航行委员会的最新计划是增加船只数量并扩展航线。”

•question:“航行委员会的计划是什么？”

The template generates text formatted as follows:
```
Answer the question with Chinese and based only on the following context:
航行委员会的最新计划是增加船只数量并扩展航线。

Question: 航行委员会的计划是什么？
```
##### b2.Prompt

Prompt plays the role of constructing and formatting input data in the code, so that the input data can be passed to the large language model (LLM) in a predetermined template format for processing. Specifically, prompt uses a defined template, inserts the actual context and question into the placeholder position in the template, and generates a complete input text for use by the large language model.

Creates a ChatPromptTemplate instance prompt based on the previously defined template template.
```
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)
```

#### c.LLM

Instantiate a large language model llm with the model name "qwen:4b".

```
from langchain_community.llms import Ollama

llm = Ollama(model="qwen:4b")
```

#### d.Chain

A processing chain chain is constructed, and its workflow is as follows:

1. Use RunnableParallel to handle two tasks in parallel: the problem of getting context from the retriever and passing the input.
2. Pass the context and question to the template prompt to generate a complete answer template.
3. Use the large language model llm to generate answers based on the template.
4. Use StrOutputParser to parse the answer into a string.

```
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
)
```

#### e.users query and LLM answer

A data model Question is defined, which contains a string root attribute __root__ to represent the input question.

Configure the processing chain to accept input of type Question.

Call the processing chain chain with the input question "航行委员会的计划是什么？" and print out the answer.
```
from langchain_core.pydantic_v1 import BaseModel

class Question(BaseModel):
    __root__: str
chain = chain.with_types(input_type=Question)
output = chain.invoke("航行委员会的计划是什么？")
print(output)
```

