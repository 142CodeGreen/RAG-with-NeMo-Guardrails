# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

#from langchain_community.chat_message_histories import ChatMessageHistory

#import gc
#import torch

import os
import gradio as gr
import shutil  # For copying files
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

#rails.documents_loaded = False

index = None
query_engine = None

#from Config.actions import init

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]


def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        kb_dir = "./Config/kb"  # Create the 'kb' directory if it doesn't exist
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
            shutil.copy2(file_path, kb_dir)

        if not documents:
            return f"No documents found in the selected files.", gr.update(interactive=False)

        # use GPU for Milvus workload
        vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024,
            collection_name="your_collection_name",
            gpu_id=0,  # Specify the GPU ID to use
            output_fields=["field1","field2"]
        )
        
        
        #vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        #index.storage_context.persist(persist_dir='./Config/kb/')   # to add sequence for rag
        
        query_engine = index.as_query_engine(similarity_top_k=20) # streaming=True)


        def test_query_engine():
            global query_engine
            if query_engine:
                try:
                    response = query_engine.query("Test Query")
                    logger.info(f"Test query response: {response.response}")
                except Exception as e:
                    logger.error(f"Query engine failed with error: {str(e)}")
            else:
                logger.error("Query engine is not initialized")

        # Call this function after initialization for testing
        test_query_engine()

        # Update app.context (This is the important line)
        #rails.documents_loaded = True

        #init(rails)
        
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files." # gr.update(interactive=True) #add interactive
    except Exception as e:
        return f"Error loading documents: {str(e)}" # gr.update(interactive=False)

from Config.actions import init
init(rails)

async def stream_response(message, history):
    if query_engine is None:
        return history + [("Please upload a file first.", None)]
        
    try:
        user_message = {"role": "user", "content": message}
        rails_response = await rails.generate(messages=[user_message]) # bot_message])
        response = history + [(message, rails_response['content'])]  # to collect gc

        # Simulate streaming by yielding intermediate steps (if possible)
        for chunk in generate_chunks_from_response(rails_response['content']):
            response[-1] = (message, chunk) 
            yield response
            await asyncio.sleep(0.05)  # Adjust delay as needed
        
        return history + [(message, rails_response['content'])]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

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
    
    #msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])   
    msg.submit(
        lambda message, history: asyncio.run(stream_response(message, history)),
        inputs=[msg, chatbot], 
        outputs=[chatbot])
        #_js="async (x) => {for await (let chunk of x) {await new Promise(r => setTimeout(r, 50)); yield chunk;}}"
    
    clear.click(lambda: None, None, chatbot, queue=False)


# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)



# use async stream_response
#async def stream_response(message, history):
#    if query_engine is None:
#        return history + [("Please upload a file first.", None)]
        
#    try:
#        user_message = {"role": "user", "content": message}
        # Assuming rails.generate() can be awaited if it's now asynchronous
#        rails_response = await rails.generate(messages=[user_message])
        
        # Create a CUDA event
 #       start_event = torch.cuda.Event(enable_timing=True)
 #       end_event = torch.cuda.Event(enable_timing=True)
 #       start_event.record()

        # Record the event after the transfer is complete (assuming rails.generate handles this)
  #      end_event.record()
  #      torch.cuda.synchronize()  # Wait for all operations in the current stream to complete

        # Wait for the event before accessing data on the GPU
  #      start_event.wait(end_event)

#        return history + [(message, rails_response['content'])]
#    except Exception as e:
#        return history + [(message, f"Error processing query: {str(e)}")]
