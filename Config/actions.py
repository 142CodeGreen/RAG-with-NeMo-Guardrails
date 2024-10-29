from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails import LLMRails
from nemoguardrails.kb.kb import KnowledgeBase
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate


def template(question, context):
    return f"""Answer user questions based on loaded documents. 
    
    {context}
    
    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Shoud not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

@action(is_system_action=True)
def rag(context: dict, llm, kb: KnowledgeBase) -> ActionResult:
    global loaded_documents
    try:
        #context_updates = {}
        message = context.get('last_user_message')
        docs = [Document(page_content=doc['content']) for doc in loaded_documents]
        chunks = kb.search_relevant_chunks(message, docs=docs)
        # chunks = kb.search_relevant_chunks(message)
        relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
        prompt = template(message, relevant_chunks)  # Pass relevant_chunks directly
        answer = llm(prompt)
        return ActionResult(return_value=answer, context_updates=context_updates)
    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates={})

#def init(app: LLMRails):
#    app.register_action(rag, "rag")
#    if not app.context.get('documents_loaded', False):  #new
#        print("Warning: Documents not loaded. Guardrails initialization delayed.")  #new
#        return  #new
#    app.register_action(rag, "rag")


# Check if documents have been loaded
#        if not context.get('documents_loaded', False):
#            return ActionResult(return_value="Documents are not yet loaded. Please upload documents first.", context_updates={})

        #input_variables = {"question": message, "context": relevant_chunks}

        #global query_engine
    
        #response = query_engine.query(message)
        #if response:
        #    relevant_chunks = response.source_nodes[0].node.text
            # Use the template function here
        #    prompt = template(message, relevant_chunks)  # Pass relevant_chunks directly
            # Use the template function here
            #context_str = "\n\n".join(chunk.text for chunk in relevant_chunks)
            #prompt = template(message, context_str)
        #    prompt_template = PromptTemplate(prompt)
        #    input_variables = {"question": message, "context": relevant_chunks}

            # Store the template for hallucination-checking
        #context_updates["_last_bot_prompt"] = prompt_template.format(
        #    **input_variables
        #)

        #print(f" RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

            # Generate answer using LlamaIndex (LLM is configured globally)
        #answer = llm(context_updates["_last_bot_prompt"])
        #return ActionResult(return_value=answer, context_updates=context_updates)
        #else:
        #    return ActionResult(return_value="No relevant information found in the loaded documents.", context_updates={})
   
