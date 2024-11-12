import os
import shutil
#import logging
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

index = None

def load_documents(file_paths):
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

        # Apply SentenceSplitter to split documents into sentences
        sentence_splitter = SentenceSplitter(
            chunk_size=1024,  # Adjust chunk_size based on your needs
            chunk_overlap=20,  # Adjust overlap for context preservation
        )
        split_documents = []
        for doc in all_documents:
            split_documents.extend(sentence_splitter.get_nodes_from_documents([doc]))

        # Convert split nodes back to Document objects if necessary
        documents_for_index =[Document(text=node.text) for node in split_documents]

        # Use local SQLite for Milvus
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents_for_index, storage_context=storage_context)

        # Sample query after indexing for verification (note: this should be done in an async context)
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        sample_query = "What is the document about?"
        try:
            sample_response = query_engine.query(sample_query)
            logger.info(f"Sample query result: {sample_query}\n{sample_response.get_formatted_sources()}")
        except Exception as e:
            logger.warning(f"Failed to perform sample query: {e}")

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
