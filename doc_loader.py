import os
import shutil
import logging
from typing import List, Tuple
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document, load_index_from_storage
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

def load_documents(file_paths: List[str]) -> Tuple[VectorStoreIndex, str]:
    global index
    if index is not None:
        logger.info("Index already exists. Skipping loading.")
        return index, "Index already loaded."

    try:
        if not file_paths:
            return None, "Error: No files selected."

        kb_dir = "./Config/kb"
        os.makedirs(kb_dir, exist_ok=True)

        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        if not documents:
            return None, "No documents found in the selected paths."

        # Move files to kb directory
        for file_path in file_paths:
            if os.path.isfile(file_path):
                shutil.copy2(file_path, kb_dir)

        # Create a Milvus vector store and storage context
        # GPU acceleration setup for Milvus
        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0  # Specify the GPU ID to use
        #)
        
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)   #CPU usage for Milvus
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Sample query for verification
        query_engine = index.as_query_engine(similarity_top_k=20)
        try:
            response = query_engine.query("What kind of wine do you have?")
            logger.info(f"Sample query result: {response.response if hasattr(response, 'response') else response}")
        except Exception as e:
            logger.error(f"Error executing sample query: {e}")

        #storage_context.persist(persist_dir="/app/storage")
        storage_context.persist(persist_dir="./storage")
        logger.info("Storage context saved to disk.")

        return index, "Documents loaded & indexed successfully"

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return None, f"Failed to index documents: {str(e)}"

def get_index() -> VectorStoreIndex:
    global index
    if index is None:
        try:
            #storage_context = StorageContext.from_defaults(persist_dir="/app/storage")
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded from storage.")
        except Exception as e:
            logger.error(f"Failed to load index from storage: {e}")
            return None
    return index
