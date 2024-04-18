import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader ,DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant


embeddings = SentenceTransformerEmbeddings(model_name ="NeuML/pubmedbert-base-embeddings")

loader = DirectoryLoader("./Data",glob="**/*.pdf",show_progress=True,loader_cls=PyPDFLoader)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=70
)

texts = text_splitter.split_documents(docs)

url ="http://localhost:6333"

qdrant =Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name ='med_db'
)
print("Med DB Successfully Created!")

