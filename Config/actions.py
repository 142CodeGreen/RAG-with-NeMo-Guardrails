from typing import Optional, Dict
import logging
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails import LLMRails
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.vector_stores.milvus import MilvusVectorStore
from doc_loader import get_index

logger = logging.getLogger(__name__)

# Set up global settings
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

@action(is_system_action=True)
async def rag(context: Dict):
    logger.info("rag() function called!")
    
    index = get_index()
    if index is None:
        logger.error("Index not available.")
        return ActionResult(
            return_value="Index not available.",
            context_updates={}
        )

    # Retrieve question from context
    question = context.get('last_user_message', '')
    logger.info(f"User question: {question}")

    try:
        # Create query engine directly from the index with global settings
        query_engine = index.as_query_engine()

        # Query the index for an answer
        logger.info(f"Querying index with: {question}")
        response = await query_engine.aquery(question)

        # Log the response metadata
        logger.info(f"Number of source nodes for the response: {len(response.source_nodes)}")

        # Extract the answer from the response
        answer = response.response

        # Update context with new information
        context_updates = {
            "relevant_chunks": "\n".join([node.text for node in response.source_nodes]),
            "history": context.get('history', []) + [(question, answer)]
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
            logger.info(f"Sample query result: {response.response}")
        except Exception as e:
            logger.error(f"Error during sample query in init: {str(e)}")
    else:
        logger.warning("No index provided to init function for sample query testing.")
