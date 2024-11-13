import os
import shutil
import logging
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import Settings

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

logger = logging.getLogger(__name__)

index = None

async def load_documents(file_paths):
    global index
    try:
        if not file_paths:
            return "Error: No files selected."

        kb_dir = "./Config/kb"
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        all_documents = []
        
        # Load all documents by path
        for file_path in file_paths:
            if os.path.isfile(file_path):  # Ensure the path points to a file
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                all_documents.extend(documents)
                shutil.copy2(file_path, kb_dir)

        if not all_documents:
            return "No documents found in the selected paths."

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(all_documents, storage_context=storage_context)

        # Sample query after indexing for verification
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        sample_query = "What is the document about?"
        sample_response = await query_engine.aquery(sample_query)
        logger.info(f"Sample query result: {sample_query}\n{sample_response.get_formatted_sources()}")

        # Save the index 
        storage_context.persist(persist_dir="./storage")
        logger.info("Storage context saved to disk.")

        # Return the index
        logger.info(f"Index created: {index}")
        return index, "Documents loaded & indexed successfully"

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return None, f"Failed to index documents: {str(e)}"

def get_index():
    global index
    if index is None:
        logger.info("No index found or it hasn't been created yet.")
        return None
    logger.info(f"Returning existing index.")
    return index
