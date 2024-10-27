from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from llama_index.core import StorageContext, load_index_from_storage
#from llama_index.llms.nvidia import NVIDIA
from nemoguardrails.kb.kb import KnowledgeBase

from nemoguardrails import LLMRails

@action(is_system_action=True)
#async def rag(context: dict, llm: NVIDIA, kb: KnowledgeBase) -> ActionResult:
async def rag(context: dict, llm, kb) -> ActionResult:
    """
    This function performs retrieval augmented generation (RAG) using LlamaIndex.
    """
    try:
        # Load the index from the 'kb' subfolder
        storage_context = StorageContext.from_defaults(persist_dir="./Config/kb")
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()

        #print("LlamaIndex knowledge base loaded successfully.") #2

        # Get the user's message from the context
        message = context.get('user_message')
        context_update = {}
        
        #if message is None:
        #    print("Error: No user_message found in context.")  # 3.
        #    return ActionResult(return_value="No user message found.", context_updates={})
        #    print(f"User input: {message}")  # 4.
        
        #llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
        Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
        #print("LLM initialized.")  # 5.

        # Search for relevant chunks using kb
        #print("Searching for relevant chunks...")  # 6.
        
        relevant_chunks = await kb.search_relevant_chunks(message)
        print(f"Relevant chunks found: {relevant_chunks}")  # 7. 
        context_updates["relevant_chunks"] = relevant_chunks
        
        #context_updates = {"retrieved_chunks": relevant_chunks}  # Store in context

        # Construct a prompt with the retrieved chunks
        prompt = f"Answer the question based on this context:\n\n{relevant_chunks}\n\nQuestion: {message}"
        print(f"Prompt constructed: {prompt}")  # 8.
        
        response = llm(prompt)
        print(f"Generated response: {response}")  # 10.

        #context_updates = {"retrieved_chunks": relevant_chunks}  # Store in context
        #print(f"Returning response with context updates: {context_updates}") #11
        
        return ActionResult(return_value=response, context_updates=context_updates)

    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")


#from utils import load_documents, query_engine
#from nemoguardrails import LLMRails
#from nemoguardrails.actions.actions import ActionResult
#from nemoguardrails.kb.kb import KnowledgeBase

#from openai import OpenAI

#async def rag(context: dict, kb: KnowledgeBase) -> ActionResult:  # Updated function signature
#    user_message = context.get("last_user_message")
#    context_updates = {}

    # For our custom RAG, we re-use the built-in retrieval
#    chunks = await kb.search_relevant_chunks(user_message)
#    relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
    # 💡 Store the chunks for fact-checking
    #context_updates["relevant_chunks"] = relevant_chunks

    # Use a custom prompt template
#    prompt = TEMPLATE.format(question=user_message, context=relevant_chunks)
    # 💡 Store the prompt for hallucination-checking
#    context_updates["_last_bot_prompt"] = prompt

#    print(f"RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

#    documents = load_documents()
    
    # Initialize the LLM and ServiceContext
#    llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

    # Put together a LlamaIndex chain
#    index = VectorStoreIndex.from_documents(documents,
 #                                            storage_context=storage_context, 
 #                                            llm_predictor=llm)

    # Get the query engine (modified)
  #  response = await query_engine.aquery(prompt)
  #  answer = response.response  # Access the 'response' attribute directly

 #   return ActionResult(return_value=answer, context_updates=context_updates)


#def init(app: LLMRails):
#    app.register_action(rag, "rag")

#async def rag(context: dict, llm, kb:KnowledgeBase) -> ActionResult: 
#    """Performs retrieval-augmented generation (RAG) to answer a user's question."""

#    print(f"kb type: {type(kb)}")  # Print the type of kb
#    print(f"kb attributes: {dir(kb)}")  # Print the attributes of kb

#    user_message = context.get("last_user_message")
#    context_updates = {}

#    if index:
#        query_eng = index.as_query_engine()  # Use the global index directly
#        chunks = query_eng.search_relevant_chunks(user_message)
#        relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
#        context_updates["relevant_chunks"] = relevant_chunks
#        context_updates["_last_bot_prompt"] = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. \n\n{relevant_chunks}\n\nQuestion: {user_message}\n\nHelpful Answer:"

#        print(f"RAG :: relevant_chunks: {relevant_chunks}")  # Log the retrieved chunks

#        response = query_eng.query(user_message, llm_predictor=llm)
#        answer = str(response)  # Convert the response to a string
#    else:
#        answer = "No documents loaded. Please load documents first."

#    return ActionResult(return_value=answer, context_updates=context_updates)

#def init(app: LLMRails):
#    """Initializes the RAG action within the LLMRails app."""
#    app.register_action(rag, "rag")
