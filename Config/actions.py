import logging
import asyncio
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from nemoguardrails import LLMRails #, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from doc_loader import load_documents, get_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

@action(is_system_action=True)
async def rag(context: Dict):
    logger.info("retrieve_relevant_chunks() function called!")
    
    index = get_index
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
        # Create query engine from index
        query_engine = index.as_query_engine()

        # Directly query the index for an answer without constructing a prompt
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

        logger.info("Returning result from retrieve_relevant_chunks()")
        return ActionResult(
            return_value=answer,
            context_updates=context_updates
        )
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunks(): {str(e)}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )
        
def init(app: LLMRails):
    # Store the index somewhere accessible, like setting it as an attribute of the app
    app.index = index
    app.register_action(rag, name="rag")
    logger.info("retrieve_relevant_chunks action registered successfully.")

    # Sample query to test the query engine
    if index:
        try:
            query_engine = index.as_query_engine()
            sample_query = "What are the documents about?"
            response = query_engine.query(sample_query)
            logger.info(f"Sample query result: {response.response}")
        except Exception as e:
            logger.error(f"Error during sample query in init: {str(e)}")
    else:
        logger.warning("No index provided to init function for sample query testing.")

