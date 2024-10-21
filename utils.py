query_engine = None  # Initialize query_engine

def load_documents(file_objs):
    global query_engine
    try:
        # ... (your existing code to load documents) ...

        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files." 
    except Exception as e:
        return f"Error loading documents: {str(e)}"
