from llama_index.core import SimpleDirectoryReader
from nemoguardrails.actions.actions import ActionResult

async def rag(context, query):
    global query_engine
    user_query = context.get("last_user_message", query)
    context_updates = {}
    
    # Access the query_engine from the context
    query_engine = context.get('query_engine') 
    
    # Perform the query
    print(f"Querying: {user_query}")
    response = query_engine.query(user_query)
    print(f"Response: {response}")
    
    context_updates["answer"] = response.response
    context_updates["relevant_chunks"] = str(response.get_formatted_sources())

    return ActionResult(return_value=response.response, context_updates=context_updates)

#def init(app):
#    app.register_action(rag, "rag")
