from llama_index.core import SimpleDirectoryReader
from nemoguardrails.actions.actions import ActionResult

async def rag(context, query):
    user_query = context.get("last_user_message", query)
    context_updates = {}
    
    # Access the query_engine from the global scope
    global query_engine
    
    # Perform the query
    print(f"Querying: {user_query}")
    response = query_engine.query(user_query)
    print(f"Response: {response}")
    
    context_updates["answer"] = response.response
    context_updates["relevant_chunks"] = str(response.get_formatted_sources())

#def init(app):
#    app.register_action(rag, "rag")
