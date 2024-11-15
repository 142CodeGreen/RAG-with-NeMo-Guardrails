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

index = load_documents([]) 

async def initialize_guardrails():
    try:
        config = RailsConfig.from_path("./Config")
        
        # Ensure index exists or has been created
        index = get_index()
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
    try:
        user_message = {"role": "user", "content": query}
        response = await rails.generate_async(messages=[user_message])
        
        if 'content' in response:  # Check if response is a direct dictionary
            # Non-streaming response
            yield history + [(query, response['content'])]
        elif hasattr(response, 'response_gen'):
            # Streaming response
            partial_response = ""
            for text in response.response_gen:
                partial_response += text
                yield history + [(query, partial_response)]
        else:
            # Handle cases where response might not have 'content' or 'response_gen'
            full_response = response if isinstance(response, str) else str(response)
            yield history + [(query, full_response)]
    except Exception as e:
        yield history + [(query, f"Error processing query: {str(e)}")]
        
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
        lambda index_result: initialize_guardrails(index_result[0]),  # Pass the index
        inputs=[load_output],  # Input is the output of load_documents
        outputs=[state, load_output]
    )
    
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
