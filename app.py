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
from Config.actions import init

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_index_on_start():
    index = get_index()
    if index is None:
        logger.warning("Index not found or could not be loaded at startup.")
    return index

index = load_index_on_start()

async def initialize_guardrails(index):
    try:
        config = RailsConfig.from_path("./Config")
        
        if index is None:
            logger.error("Index is not available during guardrails initialization.")
            return "Guardrails not initialized: No index available.", None
        
        rails = LLMRails(config, verbose=True)
        await init(rails, index)
        
        return rails, "Guardrails initialized successfully."
    except Exception as e:
        logger.error(f"Error initializing guardrails: {e}")
        return None, f"Guardrails not initialized due to error: {str(e)}"


async def stream_response(rails, query, history):
    if not rails:
        logger.error("Guardrails not initialized.")
        yield[("System", "Guardrails not initialized. Please load documents first.")]
        return

    try:
        user_message = {"role": "user", "content": query}
        result = await rails.generate_async(messages=[user_message])

        # Process the result assuming RAG was used to generate the response
        if 'content' in result:
            history.append((query, result['content']))
        elif hasattr(result, 'response'):
            history.append((query, result.response))
        else:
            history.append((query, "Unexpected response format."))
        
        yield history
    except Exception as e:
        logger.error(f"Error in stream_response: {e}")
        error_message = "An error occurred while processing your query."
        history.append((query, error_message))
        yield history

        
#async def stream_response(rails, query, history):
#    if not rails:
#        logger.error("Guardrails not initialized.")
#        yield[("System", "Guardrails not initialized. Please load documents first.")]
#        return

#    try:
#        user_message = {"role": "user", "content": query}
#        result = await rails.generate_async(messages=[user_message])

        # Depending on how the guardrails return the response, process it accordingly
#        if isinstance(result, dict):
#            if "content" in result:
#                history.append((query, result["content"]))
#            else:
#                history.append((query, str(result)))
#        elif isinstance(result, str):
#            history.append((query, result))
#        elif hasattr(result, '__iter__'):  # For streaming or chunked responses
#            full_response = ""
#            for chunk in result:
#                if isinstance(chunk, dict) and "content" in chunk:
#                    full_response += chunk["content"]
#                    history.append((query, full_response))
#                    yield history
#                else:
                    # Assuming chunk is directly the part of the response
#                    full_response += chunk
#                    history.append((query, full_response))
#                    yield history
#        else:
#            logger.error(f"Unexpected result type: {type(result)}")
#            history.append((query, "Unexpected response format."))

        # Final yield in case the response wasn't streamed
#        yield history

#    except Exception as e:
#        logger.error(f"Error in stream_response: {e}")
#        error_message = "An error occurred while processing your query."
#        history.append((query, error_message))
#        yield history

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with Guardrails")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to load documents")

    load_output = gr.Textbox(label="Load Status") 
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")
    
    # Using a State to manage shared data across callbacks
    state = gr.State(None)  # Initially None

    load_btn.click(
        load_documents,
        inputs=[file_input],
        outputs=[load_output]
    ).then(
        get_index,
        inputs=None,
        outputs=None
    ).then(
        lambda idx: initialize_guardrails(idx),
        inputs=[state],
        outputs=[state, load_output]
    )
    
    
    #load_btn.click(
    #    load_documents,
    #    inputs=[file_input],
    #    outputs=[load_output]
    #).then(
    #    lambda index_result: initialize_guardrails(index_result[0]),  # Pass the index
    #    inputs=[load_output],  # Input is the output of load_documents
    #    outputs=[state, load_output]
    #)
    
    #load_btn.click(
    #    load_documents, 
    #    inputs=[file_input], 
    #    outputs=[load_output]
    #).then(
    #    initialize_guardrails,  # This is now async
    #    None,  # No direct inputs, but it uses the global index
    #    outputs=[state, load_output]
    #)
    
    msg.submit(
        stream_response, 
        inputs=[state, msg, chatbot], 
        outputs=[chatbot]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
