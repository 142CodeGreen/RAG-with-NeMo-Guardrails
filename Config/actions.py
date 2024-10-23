# actions.py

#from llama_index.prompts import PromptTemplate

from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
#from nemoguardrails.kb.kb import KnowledgeBase

from utils import query_engine  # Import the query_engine from utils.py

#from llama_index.llms.nvidia import NVIDIA
#Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""


async def rag(context: dict, llm, kb: KnowledgeBase) -> ActionResult:
    user_message = context.get("last_user_message")
    context_updates = {}

    # For our custom RAG, we re-use the built-in retrieval
    chunks = await kb.search_relevant_chunks(user_message)
    relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
    # 💡 Store the chunks for fact-checking
    #context_updates["relevant_chunks"] = relevant_chunks

    # Use a custom prompt template
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    input_variables = {"question": user_message, "context": relevant_chunks}
    # 💡 Store the template for hallucination-checking
    context_updates["_last_bot_prompt"] = prompt_template.format(**input_variables)

    print(f"RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

    # --- Llamaindex integration without LLMPredictor and ServiceContext ---

    # Use the imported query_engine from utils.py
    #if query_engine is None:
    #    return ActionResult(
    #        return_value="Error: No documents loaded. Please load documents first.",
    #        context_updates=context_updates,
    #    )

    # Directly use the LLM object for response generation
    response = query_engine.query(user_message, llm_predictor=llm)
    answer = str(response)  # Convert the response to a string
    # --- End of Llamaindex integration ---

    return ActionResult(return_value=answer, context_updates=context_updates)


def init(app: LLMRails):
    app.register_action(rag, "rag")




#from nemoguardrails import LLMRails
#from nemoguardrails.actions.actions import ActionResult
#from nemoguardrails.kb.kb import KnowledgeBase

#from utils import load_documents

#async def rag(context: dict, llm, kb: KnowledgeBase) -> ActionResult:  # kb argument added back
#    user_message = context.get("last_user_message")
#    context_updates = {}

#    if "query_engine" not in context:  # Check if query_engine is available
#        load_documents_result, query_engine = load_documents(context.get("files"))  # Get query_engine
#        if isinstance(load_documents_result, str):  # Check for errors
#            return ActionResult(return_value=load_documents_result, context_updates=context_updates)
#        context_updates["query_engine"] = query_engine  # Store in context

    # use the query_engine from context
 #   response = context_updates["query_engine"].query(user_message)
 #   context_updates["relevant_chunks"] = response.response

    # Construct the prompt with relevant context
 #   prompt_template = f"""Use the following pieces of context to answer the question at the end.
#If you don't know the answer, just say that you don't know, don't try to make up an answer.
#Use three sentences maximum and keep the answer as concise as possible.

#{relevant_chunks}

#Question: {user_message}

#Helpful Answer:"""

    #context_updates["_last_bot_prompt"] = prompt_template
   # print(f"RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

    # Generate answer using the provided llm
  #  answer = await llm.agenerate(prompts=[prompt_template], stop=["\n"])

 #   return ActionResult(return_value=answer, context_updates=context_updates)

#def init(app: LLMRails):
#    app.register_action(rag, "rag")

