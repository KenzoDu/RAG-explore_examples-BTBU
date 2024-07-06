import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
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

file_path = "E:/llm/liulang.pdf"
raw_documents = pdf_to_documents(file_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(raw_documents)

embedding_model = OllamaEmbeddings()

texts = [doc.page_content for doc in all_splits]


persist_directory = 'E:/llm/chroma_db'


vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="RAG_chroma",
    persist_directory=persist_directory
)


vectorstore.persist() 
