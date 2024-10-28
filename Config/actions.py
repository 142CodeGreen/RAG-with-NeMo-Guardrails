from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from llama_index.core import StorageContext, load_index_from_storage
#from nemoguardrails.kb.kb import KnowledgeBase

from nemoguardrails import LLMRails

@action(is_system_action=True)
async def rag(context: dict, llm, kb) -> ActionResult:
    """
    This function performs retrieval augmented generation (RAG) using LlamaIndex.
    """
    try:
        # Load the index from the 'kb' subfolder
        #storage_context = StorageContext.from_defaults(persist_dir="./Config/kb")
        #index = load_index_from_storage(storage_context)
        #query_engine = index.as_query_engine()

        # Get the user's message from the context
        context_update = {}
        message = context.get('last_user_message')
        #context_update = {}

        #print("Searching for relevant chunks...")  # 6.
        
        relevant_chunks = await kb.search_relevant_chunks(message)
        print(f"Relevant chunks found: {relevant_chunks}")  # 7. 
        #context_updates["relevant_chunks"] = relevant_chunks
        
        prompt = f"Answer the question based on this context:\n\n{relevant_chunks}\n\nQuestion: {message}"
        response = llm(prompt)
        print(f"Generated response: {response}")  # 10.

        return ActionResult(return_value=response, context_updates=context_updates)

    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")
