from typing import Dict
from nemoguardrails.actions import action
from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
#from doc_loader import get_index
import logging
import asyncio

logger = logging.getLogger(__name__)

@action(is_system_action=True)
async def rag(context: Dict) -> ActionResult:
    """
    Retrieves relevant context from documents and generates an answer based on user query.
    
    Args:
        context (Dict): The context dictionary containing user messages and other relevant data.
    
    Returns:
        ActionResult: An object containing the return value (the answer) and any context updates.
    """
    logger.info("rag() function called!")

    index = app.index
    #index = get_index()
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
        # Create query engine with top k similar documents
        query_engine = index.as_query_engine(similarity_top_k=20)
        
        # Use query_engine to both retrieve and generate response
        response = await query_engine.aquery(question)

        # The query engine should have already processed the documents and generated a response
        # Assuming 'response' has an attribute 'response' with the generated answer
        answer = response.response

        # Update context with new information
        context_updates = {
            "relevant_chunks": response.get_formatted_sources()  # Assuming this method exists to get sources
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
    app.index = index
    app.register_action(rag, name="rag")
