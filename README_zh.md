# RAG系统-探索的例子-BTBU

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>

这个例子是对后期工作的初步探索。它使用Ollama和Langchain等框架工具制作了一个简单的程序，可以插入知识库来执行RAG问答。
### What is RAG?
LLM 可以推理广泛的主题，但他们的知识仅限于他们接受训练的特定时间点的公共数据。如果您想构建能够推理私有数据或模型截止日期后引入的数据的 AI 应用程序，则需要使用模型所需的特定信息来增强模型的知识。将适当的信息引入模型提示的过程称为检索增强生成 (RAG)。

🔷检索:用户问题用于从外部知识库检索相关上下文。为此，用户查询将被嵌入到与“向量数据库中的上下文”相同的向量空间中，然后在此空间中进行相似性搜索，以返回数据库中最相似的前 k 个数据对象到查询。

🔷增强:用户查询和检索的内容被填充到提示模板中。

🔷生成:最后，检索增强线索被输入到大预言模型中。

## 前期工作

<div align="left side">
 <img alt="ollama" height="100px" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

### Ollama*
启动并运行大型语言模型。


#### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

#### macOS

[Download](https://ollama.com/download/Ollama-darwin.zip)

#### Model
这个例子用到的是qwen:4b模型(只有2.3G)
```
pip install qwen:4b
```
### 🦜️🔗 LangChain*

#### 快速安装

With pip:
```
pip install langchain
```
With conda:
```
conda install langchain -c conda-forge
```
### PyMuPDF*
**PyMuPDF** 需要 **Python 3.8或者之后的版本**,安装呢需要 **pip** :
#### 安装
```
pip install PyMuPDF
```
### Chroma*

<p align="left side">
    <b>Chroma - the open-source embedding database</b>. <br />
    使用内存构建 Python 或 JavaScript LLM 应用程序的最快方法！
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
### Python version

它需要 **Python 3.8或之后的版本**.

这个例子运行在 Python 3.8.11在Windows 10 和 Python 3.9在Macos 14.5。


## 文件指引

📌 **Whole Flow** :全流程RAG系统代码

📌 **Two-Step Flow**:分两步实现RAG系统，可以更方便的实现

📌 **liulangdiqiu.pdf**:流浪地球 PDF样本

📌 **readme**:详细讲解-中文和英文


## 快速开始
### 整体流程图
<div align="left side">
 <img alt="Overall flow chart" height="px10" src="https://cdn-lfs-us-1.huggingface.co/repos/13/3d/133d8ca2460bf82ba2bdbe928d91a6c780364a6d0cf9005087db081cca492c02/ed22547b1538ea4fd18ea26777e14d9f7e51b3388b34d3cadf165cc37a7f63e0?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27RAG_workflow.png%3B+filename%3D%22RAG_workflow.png%22%3B&response-content-type=image%2Fpng&Expires=1720595234&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyMDU5NTIzNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzEzLzNkLzEzM2Q4Y2EyNDYwYmY4MmJhMmJkYmU5MjhkOTFhNmM3ODAzNjRhNmQwY2Y5MDA1MDg3ZGIwODFjY2E0OTJjMDIvZWQyMjU0N2IxNTM4ZWE0ZmQxOGVhMjY3NzdlMTRkOWY3ZTUxYjMzODhiMzRkM2NhZGYxNjVjYzM3YTdmNjNlMD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=LV%7E8WlQ2VBRcTeKDaFyH-ZiwsnQ76OKtGqa8vRzftKIY1qqY8vRNI7QP0H9ApdCXdBOVdMnmHMOJlq2yKnYPoQk7O5Q5O1dDcUPhbyuEvnuIIuoVNThmRxq-PCjbCuaaUuuU8jpMWW-K%7EFp2kssB2-KXX5NmLJ-qV7Xdl9jCdDCKr2JbyJN5Wach5p5rRlfh-sdB2a3vESqFI3pHvGgM7XCDbh1JmD0925Q7pvrRMXfFoBy-Ibvu31xiKnu%7EV%7EnvtVVtmtYTNrTSCI93xZ0DWhpSCOzoCP51wobl9fyoM5N9It4HXQeoxUoM8aA0wmirJLy7OtJC2yHuWnCNO3FLAg__&Key-Pair-Id=K24J24Z295AEI9">
</div>

<div align="right side">
 <img alt="overall flow chart" height="px10" src="https://s2.loli.net/2024/07/07/OUPa9gmAxZ15elW.png">
</div>
这看起来是一个极其复杂的过程，对吧？
我们分别来看一下。

### 1.前期流程图
<div align="center">
 <img alt="flow chart" height="px150" src="https://python.langchain.com/v0.2/assets/images/rag_indexing-8160f90a90a33253d0154659cf7d453f.png">
</div>

#### a.装载
Langchain提供了许多内置的文档加载器<a href="https://python.langchain.com/v0.2/docs/how_to/#document-loaders/" target="_blank">(Documents-loaders)</a>.

文档是包含文本和原始数据的字典。这里，我们使用PyMuPDF来更好地处理PDF文件中的内容。
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
#### b.分割/切块
文档分块：文本分割器将文件或文本分割成块，以防止文件信息超过LLM令牌。这一步常用的工具有RecursiveCharacterTextSplitter和CharacterTextSplitter。不同之处在于，如果块大小超过指定阈值，RecursiveCharacterTextSplitter 也会递归地将文本分割成更小的块。

🔸chunk_size: 确定分割文本时每个块中的最大字符数。它指定每个块的大小或长度。

🔸chunk_overlap: 确定分割文本时连续块之间重叠的字符数。它指定前一个块的多少内容应包含在下一个块中。

确定分割文本时连续块之间重叠的字符数。它指定前一个块的多少内容应包含在下一个块中。

Langchain还提供了多种文本分割器供你选择。<a href="https://python.langchain.com/v0.2/docs/how_to/#text-splitters">(Text Splitters)</a>.

🟢BTW:这是一个非常好用的小工具，可以清楚地看到文本分块情况。<a href="https://chunkviz.up.railway.app/">(Chunkviz)</a>.
```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(raw_documents)
```
#### c.嵌入
Embedding：然后使用Embedding将上一步划分的块文本转换为向量。LangChain提供了很多Embedding模型接口，比如OpenAI、Cohere、Hugging Face、Weaviate等，可以参考LangChain官网。<a href="https://python.langchain.com/v0.2/docs/how_to/#embedding-models">(Embedding)</a>.

这里使用简单方便的 Ollama嵌入模型。
```bash
from langchain_community.embeddings import OllamaEmbeddings

embedding_model = OllamaEmbeddings()
```
#### d.向量数据库

Store：我们将Embedding后的结果存储在VectorDB中。常见的VectorDB有Chroma、Pinecone、FAISS等，这里我使用Chroma来实现。 Chroma与LangChain集成良好，您可以直接使用LangChain的界面。<a href="https://python.langchain.com/v0.2/docs/how_to/#vector-stores">(Vector-stores)</a>.

从包含多个 Document 对象的列表 all_splits 中提取每个文档的文本内容，并将内容存储在列表 text 中。
```
texts = [doc.page_content for doc in all_splits]
```
矢量数据库持久化存储路径
```
persist_directory = 'your storage location/chroma_db'
```
这段代码使用Chroma类将文本列表文本转换为向量，将这些向量存储在名为“RAG_chroma”的集合中，存储路径为'your storage location/chroma_db'
```
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="RAG_chroma",
    persist_directory=persist_directory
)
```
### 2.后期流程图

<div align="center">
 <img alt="overall flow chart" height="px10" src="https://python.langchain.com/v0.2/assets/images/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png">
</div>

#### z.装载(可选择)

如果想随时调用数据库实现问答，必须先加载数据库的内容。

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
#### 📌提示——梳理流程

上面我们将PDF信息导入到Chroma_DB中并启动了LLM服务。接下来我们需要将整个 RAG 步骤串联起来：

1️⃣ 用户发送请求。

2️⃣ 从数据库中检索文本

3️⃣ 将请求的问题与文本检索结合起来并发送给 LLM

4️⃣ LLM根据信息来回答问题得出结果。

#### a.检索

首先我们需要创建一个Retriever，它可以根据非结构化QA返回相应的文件。 LangChain提供了很多方法并集成了第三方工具。我这里使用Vectorstore方法。其他类型可以参考 <a href="https://python.langchain.com/v0.2/docs/how_to/#retrievers/">(Retriever)</a>.

```
retriever = vectorstore.as_retriever()
```

#### b.提示词模版

提示模板有助于将用户输入和参数转换为语言模型的指令。这可用于指导模型的响应，帮助其理解上下文并生成相关且连贯的基于语言的输出。提示模板将字典作为输入，其中每个键代表提示模板中要填写的变量。提示模板输出提示值。此 PromptValue 可以传递给 LLM 或 ChatModel，也可以转换为字符串或消息列表。这个 PromptValue 存在的原因是为了方便在字符串和消息之间切换。

有几种不同类型的提示模板。<a href="https://python.langchain.com/v0.2/docs/concepts/#prompt-templates">(prompt-templates)</a>.
此示例使用 ChatPromptTemplate。

##### b1.模版
模板定义了一种结构，用于格式化输入数据（上下文和问题）并生成完整的文本模板，然后将其传递到大型语言模型（LLM）以生成最终答案。

• {context}: 该占位符将被从向量存储中检索到的相关上下文替换。

• {question}: 该占位符将被用户输入的问题替换。
```
template = """Answer the question with Chinese and based only on the following context:
{context}

Question: {question}
"""
```
📍根据我们示例中的PDF内容

•context:“航行委员会的最新计划是增加船只数量并扩展航线。”

•question:“航行委员会的计划是什么？”

该模板生成的文本格式如下：
```
Answer the question with Chinese and based only on the following context:
航行委员会的最新计划是增加船只数量并扩展航线。

Question: 航行委员会的计划是什么？
```
##### b2.提示词

Prompt在代码中起着构造和格式化输入数据的作用，以便输入数据能够以预定的模板格式传递到大语言模型（LLM）进行处理。具体来说，提示使用定义的模板，将实际上下文和问题插入到模板中的占位符位置，并生成完整的输入文本供大语言模型使用。

根据先前定义的模板创建 ChatPromptTemplate 实例提示。
```
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)
```

#### c.LLM

实例化一个大型语言模型 llm，模型名称为“qwen:4b”。

```
from langchain_community.llms import Ollama

llm = Ollama(model="qwen:4b")
```

#### d.链

构建一条处理链，其工作流程如下：

1. 使用 RunnableParallel 并行处理两个任务：从检索器获取上下文和传递输入的问题。
2. 将上下文和问题传递给模板提示以生成完整的答案模板。
3. 使用大语言模型llm根据模板生成答案。
4. 使用 StrOutputParser 将答案解析为字符串。

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

#### e.用户提问和LLM回答

定义了一个数据模型Question，其中包含一个字符串根属性__root__来表示输入问题。

配置处理链以接受问题类型的输入。

输入问题“航行委员会的计划是什么？”来调用处理链，并打印出答案。
```
from langchain_core.pydantic_v1 import BaseModel

class Question(BaseModel):
    __root__: str
chain = chain.with_types(input_type=Question)
output = chain.invoke("航行委员会的计划是什么？")
print(output)
```
### 3.结果

🔴在**Whole flow**代码运行中，使用**Nvida 1660ti GPU cuda**，运行时间为**283**秒。

结果:航行委员会的计划是为人类在未来建立新家园的过程中提供科学合理的路线、时间和空间的规划。

🔴在**Whole flow**代码运行中，使用**Apple Silicon M2max GPU**，运行时间为**26**秒。

结果:航行委员会的计划是在地球绕太阳公转的过程中进行科学研究和技术开发，以期在不久的将来能够实现人类的长期和平和发展。

你肯定会好奇为什么测试结果与原文章中的答案不一致。毫无疑问，事实确实如此。这是因为LLM回答的最终结果会受到过程中文本分割器和文本嵌入的影响，这需要对每个组件的细节进行针对性的特殊优化和调整才能获得准确的答案。
