from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase
from openai import OpenAI

from utils import index  # Import the global index

async def rag(context: dict, llm, kb:KnowledgeBase) -> ActionResult: 
    """Performs retrieval-augmented generation (RAG) to answer a user's question."""

    print(f"kb type: {type(kb)}")  # Print the type of kb
    print(f"kb attributes: {dir(kb)}")  # Print the attributes of kb

    user_message = context.get("last_user_message")
    context_updates = {}

    if index:
        query_eng = index.as_query_engine()  # Use the global index directly
        chunks = query_eng.search_relevant_chunks(user_message)
        relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
        context_updates["relevant_chunks"] = relevant_chunks
        context_updates["_last_bot_prompt"] = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. \n\n{relevant_chunks}\n\nQuestion: {user_message}\n\nHelpful Answer:"

        print(f"RAG :: relevant_chunks: {relevant_chunks}")  # Log the retrieved chunks

        response = query_eng.query(user_message, llm_predictor=llm)
        answer = str(response)  # Convert the response to a string
    else:
        answer = "No documents loaded. Please load documents first."

    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    """Initializes the RAG action within the LLMRails app."""
    app.register_action(rag, "rag")
