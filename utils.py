import os
import shutil
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)


# Initialize global variables for the index and query engine
index = None
#query_engine = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs):
    global index # query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []

        # Create the kb directory if it doesn't exist
        kb_dir = "./Config/kb"
        os.makedirs(kb_dir, exist_ok=True)
        
        for file_path in file_paths:
            try:
                # directory = os.path.dirname(file_path)
                # Move the file to the kb directory
                new_file_path = os.path.join(kb_dir, os.path.basename(file_path))
                shutil.move(file_path, new_file_path)
                
                documents.extend(SimpleDirectoryReader(input_files=[new_file_path]).load_data())
            except Exception as e:
                print(f"Error loading file {file_path}: {e}") 
                return f"Error loading file {file_path}: {e}"

        if not documents:
            return f"No documents found in the selected files."

        # Create a Milvus vector store and storage context
        # vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0,  # Specify the GPU ID to use
        #    output_fields=["field1","field2"]
        #)

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True,output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context) # Create index inside the function after documents are loaded

        # Create the query engine after the index is created
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files." # query_engine
    except Exception as e:
        return f"Error loading documents: {str(e)}"

def query_engine():
    """Returns the query engine from the global index."""
    global index
    if index:
        return index.as_query_engine(similarity_top_k=20, streaming=True)
    else:
        return None  # Or raise an exception if appropriate


