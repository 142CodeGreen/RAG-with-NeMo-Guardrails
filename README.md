# Basic RAG using NVIDIA NIM, LlamaIndex, Milvus and Gradio UI
This is a lab practice following YouTube channel of NVIDIA Developer: https://www.youtube.com/watch?v=09uDCmLzYHA&t=574s. This lab practices the following components:
1. NVIDIA NIM as foundational model- using API key to connect;
2. LlamaIndex as RAG management framework for efficient indexing and retrieval of information;
3. Using NVIDIA embeddings;
4. Milvus vector database for efficient storage and retrieval of embedding vectors. GPU-accelerated Milvus is used in this practice.
5. Gradio as chat UI which allows you to upload PDF files to complete the RAG loophole.

## Setup

1. Refer this [tutorial](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) to install requirement environments which includes:
   - Docker
   - NVIDIA Docker container tool kit & NVIDIA Driver

2. Clone the repository:
```
git clone https://github.com/142CodeGreen/Basic-RAG.git
cd Basic-RAG
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Export NVIDIA API key
```
export NVIDIA_API_KEY="your-api-key-here"
echo $NVIDIA_API_KEY
```

5. Start the GPU-accelerated Milvus container:
```
sudo docker compose up -d
```
## Usage

1. Ensure the Milvus container is running:

```
docker ps
```

2. Run the app.py:
```
python3 app.py
```

3. Open the provided URL in your web browser.

4. Upload PDF file librarie and set up the environment.

5. Process the files by clicking the "Upload PDF" button.

6. Once processing is complete, use the chat interface to query your documents.

7. If you use CPU Milvus vectorstore, replace with the following code:
```
vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True,output_fields=[])
```

## File Structure

- `app.py`: Main Streamlit application ( if you follow the step-by-step script, do not run the app.py)
- `requirements.txt`: List of Python dependencies


## This RAG can be run using a CPU. To use a GPU:
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

To utilize GPU acceleration in the vector database, ensure that:
1. Your system has a compatible NVIDIA GPU.
2. You're using the GPU-enabled version of Milvus (as shown in the setup instructions).
3. There are enough concurrent requests to justify GPU usage. GPU acceleration typically shows significant benefits under high load conditions.

