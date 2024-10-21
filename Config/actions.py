from llama_index.core import SimpleDirectoryReader
from nemoguardrails.actions.actions import ActionResult

async def rag(context, query):
    #global query_engine
    try:
        user_query = context.get("last_user_message", query)
        context_updates = {}
    
        # Access the query_engine from the global scope
        #global query_engine
    
        # Perform the query
        print(f"Querying: {user_query}")
        response = query_engine.query(user_query)
        print(f"Response: {response}")
    
        context_updates["answer"] = response.response
        context_updates["relevant_chunks"] = str(response.get_formatted_sources())
    except Exception as e:
        print(f"Error in rag action: {e}")
        return ActionResult(return_value=f"Error processing query: {e}")
        

#def init(app):
#    app.register_action(rag, "rag")
