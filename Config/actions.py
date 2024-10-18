from llama_index import ServiceContext
#from llama_index.llms import NVIDIA
from llama_index.prompts import PromptTemplate
from llama_index.output_parsers import StringOutputParser
from llama_index.knowledge_base import KnowledgeBase
from nemoguardrails import LLMRails

# Assuming NVIDIA's model can be initialized like this or through an adapter
llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

# Define your template. Note: You might need to adjust this for compatibility with LlamaIndex's prompt handling
TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

# Custom prompt template for LlamaIndex
llama_prompt = PromptTemplate(TEMPLATE)

async def rag(context: dict, service_context: ServiceContext, kb: KnowledgeBase) -> str:
    user_message = context.get("last_user_message")
    context_updates = {}

    # Assuming LlamaIndex has an async method for searching or we use an adapter
    chunks = await kb.search_relevant_chunks(user_message)
    relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
    context_updates["relevant_chunks"] = relevant_chunks

    # Format the prompt with LlamaIndex's expected input
    input_variables = {"question": user_message, "context": relevant_chunks}
    formatted_prompt = llama_prompt.format(**input_variables)
    context_updates["_last_bot_prompt"] = formatted_prompt

    print(f"💬 RAG :: prompt_template: {formatted_prompt}")

    # Use NVIDIA LLM through LlamaIndex
    parser = StringOutputParser()
    chain = llama_prompt | llm | parser
    answer = await chain.ainvoke(input_variables)

    return answer, context_updates  # Assuming we return both for compatibility with ActionResult

def init(app: LLMRails):
    # Assuming there's a way to get or create service context with NVIDIA LLM
    service_context = ServiceContext.from_defaults(llm=llm)
    app.register_action(lambda c: rag(c, service_context, app.kb), "rag")





#from typing import Optional
#from nemoguardrails.actions import action
#from llama_index.core import SimpleDirectoryReader
#from llama_index.core.llama_pack import download_llama_pack
#from llama_index.packs.recursive_retriever import RecursiveRetrieverSmallToBigPack
#from llama_index.core.base.base_query_engine import BaseQueryEngine
#from llama_index.core.base.response.schema import StreamingResponse
#import os

# Global variable to cache the query_engine
#query_engine_cache = None

#if not os.path.exists("./recursive_retriever_stb_pack"):
#    RecursiveRetrieverSmallToBigPack = download_llama_pack(
#        "RecursiveRetrieverSmallToBigPack", "./recursive_retriever_stb_pack"
#    )

#def init():
#    global query_engine_cache  # Declare to use the global variable
    # Check if the query_engine is already initialized
#    if query_engine_cache is not None:
#        print('Using cached query engine')
#        return query_engine_cache
#    try:
        # load data
#        documents = SimpleDirectoryReader("data").load_data()
#        print(f'Loaded {len(documents)} documents')

        # download and install dependencies
        #RecursiveRetrieverSmallToBigPack = download_llama_pack(
        #    "RecursiveRetrieverSmallToBigPack", "./recursive_retriever_stb_pack"
        #)

        # create the recursive_retriever_stb_pack
        recursive_retriever_stb_pack = RecursiveRetrieverSmallToBigPack(documents)

        # get the query engine
 #       query_engine_cache = recursive_retriever_stb_pack.query_engine
 #   except Exception as e:
 #       print(f"Error initiatlizing query engine:{e}")
 #       return None

#def get_query_response(query_engine: BaseQueryEngine, query: str) -> str:
#    """
#    Function to query based on the query_engine and query string passed in.
#    """
#    try:
#        response = query_engine.query(query)
#        if isinstance(response, StreamingResponse):
#            typed_response = response.get_response()
#        else:
#            typed_response = response
#        response_str = typed_response.response
#        if response_str is None:
#            return ""
#        return response_str
#    except Exception as e:
#        print(f"Error getting query response: {e}")
#        return ""

#@action(is_system_action=True)
#def user_query(context: Optional[dict] = None):
#    """
#    Function to invoke the query_engine to query user message.
#    """
#    user_message = context.get("user_message")
#    print('user_message is ', user_message)
#    query_engine = init()
#    return get_query_response(query_engine, user_message)

#@action()
#async def rag(context: dict, llm: Any, kb: Any) -> str:  # Changed to async
#    """
#    This function performs your RAG logic.
#    It takes a query string as input.
#    It should return the answer string.
#    """

#    query_engine = init()
#    if query_engine:
#        user_message = context.get("last_user_message")  # Access user message from context
#        response = get_query_response(query_engine, user_message)
#        return response
#    else:
#        return "Error initializing query engine."
