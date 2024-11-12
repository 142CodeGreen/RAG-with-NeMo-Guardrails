# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

import os
import gradio as gr
import logging
import asyncio
import torch
from nemoguardrails import LLMRails, RailsConfig
from doc_loader import load_documents, get_index
from actions import init

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rails = None

async def initialize_guardrails():
    try:
        config = RailsConfig.from_path("./Config")
        global rails
        
        # Ensure index exists or has been created
        index = get_index()
        if index is None:
            logger.error("Index is not available during guardrails initialization.")
            return "Guardrails not initialized: No index available.", None

        rails = LLMRails(config, verbose=True)
        init(rails)  # Make sure init() is called after index creation
        
        return "Guardrails initialized successfully.", None
    except Exception as e:
        logger.error(f"Error initializing guardrails: {e}")
        return f"Guardrails not initialized due to error: {str(e)}", None
        
async def stream_response(query, history):
    global rails  # Use global to access the rails variable
    if not rails:
        logger.error("Guardrails not initialized.")
        yield [("System", "Guardrails not initialized. Please load documents first.")]
        return

    try:
        user_message = {"role": "user", "content": query}
        result = await rails.generate_async(messages=[user_message])

        if isinstance(result, dict):
            if "content" in result:
                history.append((query, result["content"]))
            else:
                history.append((query, str(result)))
        else:
            if isinstance(result, str):
                history.append((query, result))
            elif hasattr(result, '__iter__'):
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        history.append((query, chunk["content"]))
                        yield history
                    else:
                        history.append((query, chunk))
                        yield history
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                history.append((query, "Unexpected response format."))

        yield history

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with Guardrails")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to load documents")

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
