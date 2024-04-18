from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_transformers import (
    LongContextReorder,
)
embeddings = SentenceTransformerEmbeddings(model_name ="NeuML/pubmedbert-base-embeddings")

url ="http://localhost:6333"
client = QdrantClient(
    url=url,
    prefer_grpc=False
)

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="med_db"
)

query="What type of patient receives what type of hormone therapy?"

return_docs =db.similarity_search_with_score(query,k=2)

for doc in return_docs:
    docs , score = doc
    print({"Score":score,"Content":docs.page_content,"meta_data":docs.metadata})
