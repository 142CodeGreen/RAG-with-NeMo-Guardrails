---
title: RAG With Guardrails
emoji: 🌖
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# RAG practice using NVIDIA NIM, NVIDIA NeMo Guardrails, LlamaIndex, Milvus Database and Gradio UI
This notebook is to practise building a RAG application with NVIDIA Nemo Guardrails under the NVIDIA Developer Program. The application includes a simple RAG that allows users to upload PDF documents and carry out Q&A with the chatbot in the context of the loaded documents. Main objectives of the app are to practice using the following key components that construct a more reliable RAG application with Nemo guardrails:

1. Selective NVIDIA NIM as a foundational LLM model- using API key to connect. meta/llama-3.1-8b-instruct is used in this practise. 
2. NVIDIA NeMo Guardrails, including input, output and dialog rails to structure proper bot response with reduced hallucination; 
3. LlamaIndex as RAG management framework for efficient indexing and retrieval of information;
4. Use of NVIDIA embeddings;
5. Milvus vector database for efficient storage and retrieval of embedding vectors. Optional to use the GPU-acceleration configuration.
6. Gradio as chat UI which allows users to upload PDF documents to set the context of Q&A.

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/RAG-with-NeMo-Guardrails.git
cd RAG-with-NeMo-Guardrails
```
2. Optional - set up a virtual environment:
```
apt-get install -y python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```
pip install --upgrade -r requirements.txt
```

4. Export API keys. NVIDIA_API_KEY is for NVIDIA NIM, while OpenAI API Key is needed for Nemo Guardrails. 
```
export NVIDIA_API_KEY="your-api-key-here"
echo $NVIDIA_API_KEY

export OPENAI_API_KEY="your-openai-key-here"
echo $OPENAI_API_KEY
```

5. Run the app.py:
```
python3 app.py
```

6. Deactivate virtual environment when finished:
```
deactivate
```

## Optional: use GPU-accelerated Milvus container:

1. Refer this [tutorial](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) to install required environments for GPU-accelerated Milvus container, including:
   - Docker
   - NVIDIA Docker container tool kit & NVIDIA Driver

To utilize GPU acceleration in the vector database, ensure that:
- Your system has a compatible NVIDIA GPU.
- You're using the GPU-enabled version of Milvus (as shown in the step 2 below).
- There are enough concurrent requests to justify GPU usage. GPU acceleration typically shows significant benefits under high load conditions.

2. It's important to note that GPU acceleration will only be used when the incoming requests are extremely high. For more detailed information on GPU indexing and search in Milvus, refer to the [official Milvus GPU Index documentation](https://milvus.io/docs/gpu_index.md).

To connect the GPU-accelerated Milvus with LlamaIndex, update the MilvusVectorStore configuration in app.py:

```
vector_store = MilvusVectorStore(
    host="127.0.0.1",
    port=19530,
    dim=1024,
    collection_name="your_collection_name",
    gpu_id=0
)
```
     
3. Upon above environment set up, start the GPU-accelerated Milvus container:
```
sudo docker compose up -d
```

4. Ensure the Milvus container is running:

```
docker ps
```

5. Run the app.py:
```
python3 -m RAG_with_NeMo_Guardrails.app
```

- Open the provided URL in your web browser.

- Upload PDF file librarie and set up the environment context.

- Process the files by clicking the "Upload PDF" button.

- Once processing is complete, use the chat interface to query your documents.


## File Structure

- Dockerfile & startup.sh files to set up NVIDIA Docker container tool kit & NVIDIA Driver for GPU-accelerated environment for Milvus Vector Database.
- Config folder to store the yaml, colang files of NeMo Guardrails, as well as a custom rag() to generate question based on loaded context. 
- `app.py`: Main application ( if you follow the step-by-step script, do not run the app.py)
- `requirements.txt`: List of application dependencies

## References: 
- NVIDIA Developer YouTube video: https://www.youtube.com/watch?v=09uDCmLzYHA&t=574s
- a NeMo Guardrail practice using LlamaIndex at https://github.com/wenqiglantz/nemo-guardrails-llamaindex-rag
- NVIDIA example of custom rag configuration for Nemo Guardrails https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/rag/custom_rag_output_rails

