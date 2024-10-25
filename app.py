# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

import os
import gradio as gr

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

from llama_index.embeddings.nvidia import NVIDIAEmbedding
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

from nemoguardrails import LLMRails, RailsConfig
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

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

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the 'kb' directory if it doesn't exist
        kb_dir = "./Config/kb"
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)
            
        # Save the index to the 'kb' subfolder
        storage_context.persist(persist_dir="./Config/kb")  

        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

async def stream_response(message, history):
    global query_engine  # You still need the query_engine for initial context
    if query_engine is None:
        yield history + [("Please upload a file first.", None)]
        return

    try:
        # 1. Get an initial response
        initial_response = query_engine.query(message) 

        # 2. Initialize partial_response 
        partial_response = ""

        # 3. Stream the initial response for the first few tokens
        for text_chunk in initial_response.response_gen:  
            partial_response += text_chunk
            
            # Apply Nemo Guardrails to the partial response
            user_message = {"role": "user", "content": message}
            bot_message = {"role": "bot", "content": partial_response}
            rails_response = rails.generate(messages=[user_message, bot_message])
            yield history + [(message, rails_response['content'])] 

            # Break early to avoid streaming the entire initial response
            if len(partial_response) > 100:  # Adjust the threshold as needed
                break 

        # 4. call rails.generate with stream=True to stream the rest
        user_message = {"role": "user", "content": message}
        rails_response_gen = rails.generate(messages=[user_message], stream=True)

        # 5. Stream the remaining response from rails.generate
        async for rails_response in rails_response_gen:  # Use async for to handle the asynchronous generator
            yield history + [(message, rails_response['content'])]
            
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

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(await stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use submit button instead of msg
    clear.click(lambda: None, None, chatbot, queue=False)

    # Initialize and register the rag action
    #setup_rails_actions()
    #rails.register_action(rag, "rag")  # Register the action with rails

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

