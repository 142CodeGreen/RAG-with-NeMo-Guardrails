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

index = None
query_engine = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

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

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Save the index to the 'kb' subfolder
        #storage_context.persist(persist_dir="./Config/kb")  

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
    global query_engine  # You still need the query_engine for initial context
    if query_engine is None:
        return history + [("Please upload a file first.",None)]
        #return history + [{"role": "user", "content": message}, {"role": "bot", "content": "Please upload a file first.", None}]
        

    try:
        full_response = query_engine.query(message)
        #for chunk in query_engine.query(message):
        #    full_response += chunk.response # accumulate response chunks    
        #    yield history + [{"role": "user", "content": message}, {"role": "bot", "content": full_response}]

        # Apply Nemo Guardrails to the chunk
        user_message = {"role": "user", "content": message}
        bot_message = {"role": "bot", "content": full_response.response}
        rails_response = rails.generate(messages=[user_message, bot_message], context={"knowledge": full_response.response})  # Include context
        return history + [{"role": "user", "content": message}, {"role": "bot", "content": rails_response['content']}]  
        
        #        yield {"role": "user", "content": rails_response['content']}

    except Exception as e:
        return history + [{"role": "user", "content": message}, {"role": "bot", "content": f"Error processing query: {str(e)}"}]


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for PDF Files")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load PDF Documents only")

    load_output = gr.Textbox(label="Load Status")
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(label="Enter your question",interactive=True)
    clear = gr.Button("Clear")

    init(rails)

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use submit button instead of msg
    clear.click(lambda: None, None, chatbot, queue=False)

    
# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)

# Import from utils.py
#from utils import load_documents

# Import actions 
#from Config.actions import init  # Import the init function

#import getpass
#import os
#import gradio as gr
#from openai import OpenAI


# Set the environment

#from llama_index.core import Settings
#from llama_index.llms.nvidia import NVIDIA
#Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

#from llama_index.embeddings.nvidia import NVIDIAEmbedding
#Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

#from llama_index.vector_stores.milvus import MilvusVectorStore

#from llama_index.core.node_parser import SentenceSplitter
#Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Import Nemo modules

#from nemoguardrails import LLMRails, RailsConfig

# Define a RailsConfig object
#config = RailsConfig.from_path("./Config")
#rails = LLMRails(config)

#init(rails)

#rails.register_action(rag, "rag")

#query_engine = None

# Initialize global variables for the index and query engine

# Function to get file names from file objects

# Function to load documents and create the index

#def stream_response(message, history):
#    #global query_engine
    #if query_engine is None:
#        #query_engine = init()  # Assuming init() is defined in actions.py
        #if query_engine is None:
        #return history + [("Failed to initialize query engine. Please check your setup.", None)]
        
#    try:
#        user_message = {"role": "user", "content": message}
#        rails_response = rails.generate(messages=[user_message]) # context={"query": message}) No context
#        yield history + [(message, rails_response['content'])]
#    except Exception as e:
#        yield history + [(message, f"Error processing query: {str(e)}")]

