from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase

from utils import load_documents
#from llama_index.core import Settings

async def rag(context: dict, llm, kb: KnowledgeBase) -> ActionResult:  # kb argument added back
    user_message = context.get("last_user_message")
    context_updates = {}

    if "query_engine" not in context:  # Check if query_engine is available
        load_documents_result, query_engine = load_documents(context.get("files"))  # Get query_engine
        if isinstance(load_documents_result, str):  # Check for errors
            return ActionResult(return_value=load_documents_result, context_updates=context_updates)
        context_updates["query_engine"] = query_engine  # Store in context

    # use the query_engine from context
    response = context_updates["query_engine"].query(user_message)
    context_updates["relevant_chunks"] = response.response

    # Construct the prompt with relevant context
    prompt_template = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{relevant_chunks}

Question: {user_message}

Helpful Answer:"""

    context_updates["_last_bot_prompt"] = prompt_template
    print(f"RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

    # Generate answer using the provided llm
    answer = await llm.agenerate(prompts=[prompt_template], stop=["\n"])

    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")


#from utils import query_engine, llm, kb_dir

#from nemoguardrails import LLMRails
#from nemoguardrails.actions.actions import ActionResult
#from nemoguardrails.kb.kb import KnowledgeBase

#async def rag(context: dict, kb: KnowledgeBase) -> ActionResult:
#    user_message = context.get("last_user_message")
#    context_updates = {}

    # LlamaIndex retrieval
#    response = query_engine.query(user_message)
#    relevant_chunks = response.response
#    context_updates["relevant_chunks"] = relevant_chunks

    # Construct the prompt directly (replace with your actual prompt structure)
#    prompt = f"Use the following context to answer the question:\n\n{relevant_chunks}\n\nQuestion: {user_message}\n\nAnswer:"
#    context_updates["_last_bot_prompt"] = prompt  

 #   print(f" RAG :: prompt: {prompt}")

    # Generate the answer using your LLM (from utils.py)
 #   answer = await llm.agenerate(prompts=[prompt], temperature=0.1) 
 #   answer = answer.generations[0][0].text

  #  return ActionResult(return_value=answer, context_updates=context_updates)

#def init(app: LLMRails):
#    app.register_action(rag, "rag")




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
