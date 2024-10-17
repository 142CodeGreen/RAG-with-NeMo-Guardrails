# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

import getpass
import os
import gradio as gr
from openai import OpenAI


# from langchain.memory import ChatMessageHistory  # Old import
# from langchain_community.chat_message_histories import ChatMessageHistory

# Set the environment

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

from llama_index.embeddings.nvidia import NVIDIAEmbedding
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Import Nemo modules

from nemoguardrails import LLMRails, RailsConfig

# Define a RailsConfig object
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

# Initialize global variables for the index and query engine
index = None
query_engine = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs):
    global index, query_engine, rails
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

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
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

# Function to handle chat interactions
def chat(message,history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.",None)]
    try:
        #modification for nemo guardrails ( next three rows)
        user_message = {"role":"user","content":message}
        response = rails.generate(messages=[user_message])
        return history + [(message,response['content'])]
    except Exception as e:
        return history + [(message,f"Error processing query: {str(e)}")]

def stream_response(message, history):
    global query_engine, rails  # Assuming rails is what we're using globally now
    if query_engine is None:
        yield history + [("Please upload a file first.", None)]
        return

    try:
        user_message = {"role": "user", "content": message}
        full_response = rails.generate(messages=[user_message])
        partial_response = ""
        for char in full_response['content']:
            partial_response += char
            yield history + [(message, partial_response)]
        # Yield one last time to ensure the final state is shown
        yield history + [(message, full_response['content'])]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
  gr.Markdown("# RAG Chatbot for PDF Files")

  with gr.Row():
      file_input = gr.File(label="Select files to upload", file_count="multiple")
      load_btn = gr.Button("Load PDF Documents only")

  load_output = gr.Textbox(label="Load Status")

  chatbot = gr.Chatbot()
  msg = gr.Textbox(label="Enter your question",interactive=True)
  clear = gr.Button("Clear")

# Set up event handler (Event handlers should be defined within the 'with gr.Blocks() as demo:' block)
  load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
  msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use chat and submit button instead of msg
  clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
