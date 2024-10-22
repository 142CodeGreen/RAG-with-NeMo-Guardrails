# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

# Import from utils.py
from utils import query_engine, load_documents

# Import actions 
#from Config.actions import rag  # Import the init function

#import getpass
import os
import gradio as gr
from openai import OpenAI


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

from nemoguardrails import LLMRails, RailsConfig

# Define a RailsConfig object
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

# Initialize global variables for the index and query engine

# Function to get file names from file objects

# Function to load documents and create the index

def chat(message, history):
    global query_engine
    if query_engine is None:
        query_engine = init()  # Assuming init() is defined in actions.py
        if query_engine is None:
            return history + [("Failed to initialize query engine. Please check your setup.", None)]
    try:
        # update for rails
        user_message = {"role":"user","content":message}
        response = rails.generate(messages=[user_message])
        return history + [(message,response['content'])]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

def stream_response(message, history):
    global query_engine
    if query_engine is None:
        query_engine = init()  # Assuming init() is defined in actions.py
        if query_engine is None:
            return history + [("Failed to initialize query engine. Please check your setup.", None)]
    try:
        user_message = {"role": "user", "content": message}
        rails_response = rails.generate(messages=[user_message], context={"query": message})  # No context
        yield history + [(message, rails_response['content'])]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]


#def stream_response(message, history):
#    global query_engine
    #if query_engine is None:
    #    yield history + [("Query engine not initialized. Please load documents first.", None)]
    #    return
    #    query_engine = init()  # Make sure init() is available here
    #    if query_engine is None:
    #        yield history + [("Failed to initialize query engine. Please check your setup.", None)]
    #        return
#    try:
        #Add query engine to context
        context = {"query_engine": query_engine}
        user_message = {"role": "user", "content": message}
        rails_response = rails.generate(messages=[user_message], context=context)
#        yield history + [(message, rails_response['content'])]
#    except Exception as e:
#        yield history + [(message, f"Error processing query: {str(e)}")]

# Import actions 
from Config.actions import rag  # Import the init function

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

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use submit button instead of msg
    clear.click(lambda: None, None, chatbot, queue=False)

    # Initialize and register the rag action
    rails.register_action(rag, "rag")  # Register the action with rails

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
