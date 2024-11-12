# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

#from langchain_community.chat_message_histories import ChatMessageHistory

import os
import gradio as gr
import logging
import asyncio
import torch
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from nemoguardrails import LLMRails, RailsConfig
from doc_loader import get_index


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

async def rag(message, history):
    index = get_index()
        if index is None:
            logger.error("Index is not available during guardrails initialization.")
            return "Index not available, pls upload documents.", None
        
    query_engine = index.as_query_engine()
    
    try:
        user_message = {"role": "user", "content": message}
        result = await rails.generate_async(messages=[user_message]) 
        
        for chunk in result:
            if chunk:  # Check if the chunk has content
                history.append((message, chunk)) 
                yield history

        # If you want to mark the end of the response
        history.append((None, "End of response.")) 
        yield history

    except Exception as e:
        history.append((message, f"Error processing query: {str(e)}"))
        yield history


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for PDF Files")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load PDF Documents only")

    load_output = gr.Textbox(label="Load Status") # interactive=False) 
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    #msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)


# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
