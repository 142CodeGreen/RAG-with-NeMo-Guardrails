from utils import query_engine, llm, kb_dir

from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase

async def rag(context: dict, kb: KnowledgeBase) -> ActionResult:
    user_message = context.get("last_user_message")
    context_updates = {}

    # LlamaIndex retrieval
    response = query_engine.query(user_message)
    relevant_chunks = response.response
    context_updates["relevant_chunks"] = relevant_chunks

    # Construct the prompt directly (replace with your actual prompt structure)
    prompt = f"Use the following context to answer the question:\n\n{relevant_chunks}\n\nQuestion: {user_message}\n\nAnswer:"
    context_updates["_last_bot_prompt"] = prompt  

    print(f" RAG :: prompt: {prompt}")

    # Generate the answer using your LLM (from utils.py)
    answer = await llm.agenerate(prompts=[prompt], temperature=0.1) 
    answer = answer.generations[0][0].text

    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")




#from llama_index.core import SimpleDirectoryReader
#from nemoguardrails.actions.actions import ActionResult

#async def rag(query):
#    from utils import query_engine
#    try:
#        print(f"Querying: {query}")
#        response = query_engine.query(query)
#        print(f"Response: {response}")

#        context_updates = {
#            "answer": response.response,
#            "relevant_chunks": str(response.get_formatted_sources())
#        }
#        return ActionResult(context_updates=context_updates)
#    except Exception as e:
#        print(f"Error in rag action: {e}")
#        return ActionResult(return_value=f"Error processing query: {e}")

        #user_query = query
        #context_updates = {}

        # Perform the query
        #print(f"Querying: {user_query}")
        #response = query_engine.query(user_query)
        #print(f"Response: {response}")

        #context_updates["answer"] = response.response
        #context_updates["relevant_chunks"] = str(response.get_formatted_sources())
        #return ActionResult(context_updates=context_updates)
    #except Exception as e:
        #print(f"Error in rag action: {e}")
        #return ActionResult(return_value=f"Error processing query: {e}")

#def init(app):
#    app.register_action(rag, "rag")
