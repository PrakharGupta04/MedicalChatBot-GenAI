from src.helper import load_pdf_file, text_splits, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data='Data/')
text_chunks=text_splits(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",#as it is free and give us upto 2gb for more we need ti buy sub.
        region="us-east-1"
    )
)



docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, 
    index_name=index_name,
    embedding=embeddings,
    
)