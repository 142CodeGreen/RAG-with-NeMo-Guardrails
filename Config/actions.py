from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from nemoguardrails import action, LLMRails, ActionResult

# Import the vector_store from app.py
from app import vector_store 

@action(is_system_action=True)
async def rag(message: str) -> ActionResult:
    """
    This function uses LlamaIndex to answer questions based on the files uploaded by the user.
    """
    context_updates = {}
    try:
        # Use the existing vector_store from app.py
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load the index from the 'kb' subfolder
        index = load_index_from_storage(storage_context, persist_dir="./Config/kb")
        query_engine = index.as_query_engine(similarity_top_k=20)

        # Get the answer from the query engine
        response = query_engine.query(message)

        # You can add context updates here if needed
        # For example, to add the retrieved source nodes to the context:
        # context_updates = {"source_nodes": response.source_nodes} 

        return ActionResult(return_value=response.response, context_updates=context_updates)

    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")
