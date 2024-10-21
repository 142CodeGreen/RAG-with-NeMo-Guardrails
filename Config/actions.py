from llama_index.core import SimpleDirectoryReader
from nemoguardrails.actions.actions import ActionResult

async def rag(query):
    from utils import query_engine
    try:
        user_query = query
        context_updates = {}

        # Perform the query
        print(f"Querying: {user_query}")
        response = query_engine.query(user_query)
        print(f"Response: {response}")

        context_updates["answer"] = response.response
        context_updates["relevant_chunks"] = str(response.get_formatted_sources())
        return ActionResult(context_updates=context_updates)
    except Exception as e:
        print(f"Error in rag action: {e}")
        return ActionResult(return_value=f"Error processing query: {e}")

#def init(app):
#    app.register_action(rag, "rag")
