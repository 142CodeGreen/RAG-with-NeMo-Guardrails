from llama_index.core import SimpleDirectoryReader
from nemoguardrails.actions.actions import ActionResult

async def rag(context, query):
    #global query_engine
    try:
        user_query = context.get("last_user_message", query)
        context_updates = {}

        # Access query_engine from context
         query_engine = context.get("query_engine") 
         if query_engine is None:
             return ActionResult(return_value="Error: Query engine not initialized.")
    
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
