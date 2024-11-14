from typing import Dict, Optional
from nemoguardrails.actions import action
from nemoguardrails import LLMRails
from nemoguardrails.actions import ActionResult
from llama_index.core.retrievers import BaseRetriever
from doc_loader import get_index  # Importing get_index from doc_loader.py
import logging

logger = logging.getLogger(__name__)

@action(is_system_action=True)
async def rag(context: Dict, llm: LLMRails):
    logger.info("rag() function called!")
    
    index = get_index()
    if index is None:
        logger.error("Index not available.")
        return ActionResult(
            return_value="Index not available. Please ensure documents are loaded.",
            context_updates={}
        )

    # Retrieve question from context
    question = context.get('last_user_message', '')
    logger.info(f"User question: {question}")

    try:
        # Use the index to get relevant documents or chunks
        retriever = index.as_retriever(similarity_top_k=3)
        
        # Retrieve relevant chunks directly
        nodes = retriever.retrieve(question)
        relevant_chunks = "\n".join([node.text for node in nodes])

        # Define the custom prompt template for RAG
        TEMPLATE = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""

        # Format the prompt
        formatted_prompt = TEMPLATE.format(context=relevant_chunks, question=question)

        # Use LLMRails with NVIDIA LLM to generate the answer
        answer = await llm.generate_async(formatted_prompt)

        # Update context with new information
        context_updates = {
            "relevant_chunks": relevant_chunks,
            "_last_bot_prompt": formatted_prompt
        }

        logger.info("Returning result from rag()")
        return ActionResult(
            return_value=answer,
            context_updates=context_updates
        )
    except Exception as e:
        logger.error(f"Error in rag(): {str(e)}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )

async def init(app: LLMRails):
    app.index = get_index()
    app.register_action(rag, "rag")
    logger.info("rag action registered successfully.")

    # Sample query to test the query engine
    if app.index:
        try:
            query_engine = app.index.as_query_engine()
            sample_query = "What are the documents about?"
            response = await query_engine.aquery(sample_query)
            logger.info(f"Sample query sources: {response.get_formatted_sources()}")
        except Exception as e:
            logger.error("Error during sample query in init:", exc_info=True)
    else:
        logger.warning("No index provided to init function for sample query testing.")
