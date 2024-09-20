# Basic RAG using NVIDIA NIM, LlamaIndex, Milvus and Gradio UI
This is a lab practice following YouTube channel of NVIDIA Developer: https://www.youtube.com/watch?v=09uDCmLzYHA&t=574s. This lab practices the following components:
1. NVIDIA NIM as foundational model- using API key to connect;
2. LlamaIndex as RAG management framework for efficient indexing and retrieval of information;
3. Using NVIDIA embeddings;
4. Milvus vector database for efficient storage and retrieval of embedding vectors.
5. You can upload your own enterprise data in PDF format to complete the RAG loophole.  Q&A chatbot UI uses Gradio simple format

## Setup

1. Clone the repository:
```
git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/multimodal_rag
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Import necessary librarie and set up the environment
```
import getpass
import os
import gradio as gr
os.environ['NVIDIA_API_KEY'] = "your api key"

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    raise ValueError("Please set the NVIDIA_API_KEY environment variable.")

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

from llama_index.embeddings.nvidia import NVIDIAEmbedding
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=500)
```

4. Initialize global variables for the index and query engine
```
index = None
query_engine = None
```

5.Function to get file names from file objects
```
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]
```

6. Function to load documents and create the index
```
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

        # Create a Milvus vector store and storage context
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True,output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context) # Create index inside the function after documents are loaded

        # Create the query engine after the index is created
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"
```

7. Function to handle chat interactions
```
def chat(message,history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.",None)]
    try:
        response = query_engine.query(message)
        return history + [(message,response)]
    except Exception as e:
        return history + [(message,f"Error processing query: {str(e)}")]

# Function to stream responses
def stream_response(message,history):
    global query_engine
    if query_engine is None:
        yield history + [("Please upload a file first.",None)]
        return

    try:
        response = query_engine.query(message)
        partial_response = ""
        for text in response.response_gen:
            partial_response += text
            yield history + [(message,partial_response)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]
```

8. Create the Gradio interface
```
with gr.Blocks() as demo:
  gr.Markdown("# RAG Q&A Chatbot Testing")

  with gr.Row():
      file_input = gr.File(label="Select files to upload", file_count="multiple")
      load_btn = gr.Button("Load PDF Documents only")

  load_output = gr.Textbox(label="Load Status")

  chatbot = gr.Chatbot()
  msg = gr.Textbox(label="Enter your question",interactive=True)
  clear = gr.Button("Clear")

# Set up event handler
  load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
  msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use submit button instead of msg
  msg.submit(lambda: "", outputs= [msg]) # Use submit button and message instead of msg
  clear.click(lambda: None, None, chatbot, queue=False)
```

## Usage

Launch the Gradio interface
```
if __name__ == "__main__":
    demo.queue().launch(share=True)
```

## Alternatively, you can simply run the app.py which has compliled all scripts. 

```
python run app.py
```

Once the process is complted,open the provided URL in your web browser.

Upload your PDF file and start to query the chatbot. 

## File Structure

- `app.py`: Main Streamlit application ( if you follow the step-by-step script, do not run the app.py)
- `requirements.txt`: List of Python dependencies


## GPU Acceleration for Vector Search
To utilize GPU acceleration in the vector database, ensure that:
1. Your system has a compatible NVIDIA GPU.
2. You're using the GPU-enabled version of Milvus (as shown in the setup instructions).
3. There are enough concurrent requests to justify GPU usage. GPU acceleration typically shows significant benefits under high load conditions.

It's important to note that GPU acceleration will only be used when the incoming requests are extremely high. For more detailed information on GPU indexing and search in Milvus, refer to the [official Milvus GPU Index documentation](https://milvus.io/docs/gpu_index.md).

To connect the GPU-accelerated Milvus with LlamaIndex, update the MilvusVectorStore configuration in app.py:
```
vector_store = MilvusVectorStore(
    host="127.0.0.1",
    port=19530,
    dim=1024,
    collection_name="your_collection_name",
    gpu_id=0  # Specify the GPU ID to use
)
```
