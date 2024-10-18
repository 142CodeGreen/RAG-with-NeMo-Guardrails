# from llama_index import ServiceContext

# a directory of documents
#documents = SimpleDirectoryReader('data').load_data()
documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

# Service context setup with LLM
#from llama_index.llms import NVIDIA  # Example, use whatever LLM you have access to
#service_context = ServiceContext.from_defaults(llm=NVIDIA(model="meta/llama-3.1-8b-instruct"))

# Create an index
#index = VectorStoreIndex.from_documents(documents, service_context=service_context)
#index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Create a query engine
#query_engine = index.as_query_engine()
#query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

async def rag(context, query):
    user_query = context.get("last_user_message", query)
    context_updates = {}
    
    # Perform the query
    response = query_engine.query(user_query)
    
    # response.response contains the generated answer with context
    context_updates["answer"] = response.response
    context_updates["relevant_chunks"] = str(response.get_formatted_sources())  # If want to see what was used for the answer

    return ActionResult(return_value=response.response, context_updates=context_updates)

# Registering the action
def init(app):
    app.register_action(rag, "rag")


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
