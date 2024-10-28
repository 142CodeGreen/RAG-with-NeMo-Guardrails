# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

import os
import gradio as gr
import shutil  # For copying files
#import asyncio

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

from llama_index.embeddings.nvidia import NVIDIAEmbedding
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

from nemoguardrails import LLMRails, RailsConfig
#from nemoguardrails.streaming import StreamingHandler

config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

# Import the init function from actions.py
from Config.actions import init
init(rails)

index = None
query_engine = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def initialize_kb(documents):
    """Initialize the knowledge base with the provided documents."""
    vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        # Create the 'kb' directory if it doesn't exist
        kb_dir = "./Config/kb"
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

            # Copy the PDF file to the kb directory
            shutil.copy2(file_path, kb_dir) 

        if not documents:
            return f"No documents found in the selected files."

        #vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        #storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Save the index to the 'kb' subfolder
        #storage_context.persist(persist_dir="./Config/kb")  
        
        index = initialize_kb(documents)
        query_engine = index.as_query_engine(similarity_top_k=20) # streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

def stream_response(message, history):
    """Handle chat interactions."""
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.", None)]
        
    try:
        response = query_engine.query(message)
        response_text = response.response
        
        # Using Nemo Guardrails to process the response
        user_message = {"role": "user", "content": message}
        bot_message = {"role": "bot", "content": response.response}
        rails_response = rails.generate(messages=[user_message, bot_message], 
                                        context={"kb": response})

        return history + [(message, rails_response['content'])]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for PDF Files")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load PDF Documents only")

    load_output = gr.Textbox(label="Load Status")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)


# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
