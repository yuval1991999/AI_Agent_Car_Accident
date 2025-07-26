from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

loader = TextLoader(FILE_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
split_docs = splitter.split_documents(docs)

# from zep_python.langchain import ZepLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = ZepLoader(
#     api_url="https://api.getzep.com",
#     project_id="4953206b-f356-47ab-beb8-6de4a52243a4",
#     api_key="your_zep_api_key"
# )

# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_docs = splitter.split_documents(docs)
