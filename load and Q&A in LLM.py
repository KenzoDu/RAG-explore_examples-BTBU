from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


chroma_db_path = "your storage location/chroma_db"

embedding_model = OllamaEmbeddings()

vectorstore = Chroma.from_texts(
    embedding=embedding_model,
    texts=chroma_db_path,
    collection_name="RAG_chroma"
)
retriever = vectorstore.as_retriever()

template = """Answer the question with Chinese and based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = Ollama(model="qwen:4b")

chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
)


class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

output = chain.invoke("航行委员会的计划是什么？")
print(output)
