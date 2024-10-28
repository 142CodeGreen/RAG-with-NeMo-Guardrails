from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from llama_index.core import StorageContext, load_index_from_storage
from nemoguardrails import LLMRails

def load_kb_from_storage():
    """Load the knowledge base from storage."""
    vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    storage_context = StorageContext.from_defaults(persist_dir="./Config/kb")
    index = load_index_from_storage(storage_context)
    return index

def template(question, context):
    return f"""Use the following pieces of context to answer the question at the end.
    
    {context}
    
    1.You only answer the USER QUESTION using the CONTEXT INFORMATION. 
    2. You do not make up a story. 
    3. Keep your answer as concise as possible.
    4. Shoud not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

@action(is_system_action=True)
def rag(context: dict, llm) -> ActionResult:
    
    try:
        context_update = {}
        message = context.get('last_user_message')
        
        kb = load_kb_from_storage()
        query_engine = kb.as_query_engine(similarity_top_k=20)
        response = query_engine.query(message)
        relevant_chunks = response.source_nodes
        context_updates["relevant_chunks"] = relevant_chunks

        # Construct the prompt for the LLM
        prompt = f"Answer the question based on this context:\n\n{relevant_chunks}\n\nQuestion: {message}"
        generated_response = llm.(prompt)

        context_updates = {
            "relevant_chunks": relevant_chunks,
            "_last_bot_prompt": prompt
        }

        return ActionResult(return_value=generated_response, context_updates=context_updates)

    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates={})

def init(app: LLMRails):
    """Initialize the RAG action with the LLMRails instance."""
    app.register_action(rag, "rag")
