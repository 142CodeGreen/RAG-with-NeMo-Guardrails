from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from llama_index.core import StorageContext, load_index_from_storage
from nemoguardrails import LLMRails

def load_kb_from_storage():
    """Load the knowledge base from storage."""
    storage_context = StorageContext.from_defaults(persist_dir="./Config/kb")
    index = load_index_from_storage(storage_context)
    return index

@action(is_system_action=True)
def rag(context: dict, llm) -> ActionResult:
    """
    Retrieve and generate a response based on the user's message using RAG.
    
    :param context: Dictionary containing the user's last message
    :param llm: Language model instance for generating responses
    :return: ActionResult with the generated response and updates to context
    """
    try:
        message = context.get('last_user_message')
        kb = load_kb_from_storage()
        # Query the knowledge base for relevant information
        query_engine = kb.as_query_engine(similarity_top_k=20)
        response = query_engine.aquery(message)
        relevant_chunks = response.source_nodes

        # Construct the prompt for the LLM
        prompt = f"Answer the question based on this context:\n\n{relevant_chunks}\n\nQuestion: {message}"
        generated_response = llm.invoke(prompt).content

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
