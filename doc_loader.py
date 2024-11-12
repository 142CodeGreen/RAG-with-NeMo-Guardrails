from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter

import shutil  # For copying files
import logging

index = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index
    try:
        if not file_objs:
            return "Error: No files selected."

        kb_dir = "./Config/kb"  # Create the 'kb' directory if it doesn't exist
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
            shutil.copy2(file_path, kb_dir)

        if not documents:
            return f"No documents found in the selected files.", gr.update(interactive=False)

        # use GPU for Milvus workload
        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0
        #)  # Specify the GPU ID to use
            #output_fields=["field1","field2"]

        
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Sample query after indexing for verification
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        sample_query = "What is the document about?"
        sample_response = query_engine.aquery(sample_query)
        logger.info(f"Sample query result: {sample_query}\n{sample_response.get_formatted_sources()}")

        # Save the index 
        storage_context.persist(persist_dir="./storage")  # Choose a directory to save to
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
