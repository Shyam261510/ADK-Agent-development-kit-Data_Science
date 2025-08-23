"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
"""

import logging
from google.adk.tools.tool_context import ToolContext
from vertexai import rag
from ..config import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_TOP_K
from .utils import check_corpus_exists, get_corpus_resource_name


def rag_query(corpus_name: str, query: str, tool_context: ToolContext) -> dict:
    """
    Query a Vertex AI RAG corpus with user questions and retrieve relevant information.

    Args:
        corpus_name (str): The name of the corpus to query.
                           Preferably use the resource_name from list_corpus results.
        query (str): The text query to search for in the corpus.
        tool_context (ToolContext): The tool context.

    Returns:
        dict: The query results and status.
    """

    try:
        corpus_name
        print("\n========== 🔍 Starting RAG Query ==========")
        print(f"📂 Corpus Name: {corpus_name}")
        print(f"💬 Query: {query}")

        # --- Check if corpus exists ---
        print("🔍 Checking if corpus exists...")
        if not check_corpus_exists(corpus_name=corpus_name, tool_context=tool_context):
            print(f"❌ Corpus '{corpus_name}' not found.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "query": query,
                "corpus_name": corpus_name,
            }
        print(f"✅ Corpus '{corpus_name}' found.")

        # --- Get the full corpus resource name ---
        full_corpus_name = get_corpus_resource_name(corpus_name=corpus_name)
        print(f"📌 Full Corpus Resource Name: {full_corpus_name}")

        # --- Configure retrieval parameters ---
        print("⚙️ Configuring retrieval parameters...")
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )
        print(
            f"🔧 Retrieval Config: Top K = {DEFAULT_TOP_K}, Distance Threshold = {DEFAULT_DISTANCE_THRESHOLD}"
        )

        # --- Perform the query ---
        print("🚀 Performing retrieval query...")
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=full_corpus_name)],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )
        print("📨 Query executed successfully. Processing results...")

        # --- Process the response into a usable format ---
        results = []
        if hasattr(response, "contexts") and response.contexts:
            for context_group in response.contexts.contexts:
                result = {
                    "source_uri": getattr(context_group, "source_uri", ""),
                    "source_name": getattr(context_group, "source_display_name", ""),
                    "text": getattr(context_group, "text", ""),
                    "score": getattr(context_group, "score", 0.0),
                }
                results.append(result)

        # --- Handle empty results ---
        if not results:
            print(
                f"⚠️ No relevant results found in corpus '{corpus_name}' for query '{query}'."
            )
            return {
                "status": "warning",
                "message": f"No results found in corpus '{corpus_name}' for query: '{query}'",
                "query": query,
                "corpus_name": corpus_name,
                "results": [],
                "results_count": 0,
            }

        # --- Return successful results ---
        print(f"🎉 Query successful! Retrieved {len(results)} result(s).")
        print("=========================================================\n")
        return {
            "status": "success",
            "message": f"Successfully queried corpus '{corpus_name}'",
            "query": query,
            "corpus_name": corpus_name,
            "results": results,
            "results_count": len(results),
        }

    except Exception as e:
        error_msg = f"❌ ERROR: Failed to query corpus '{corpus_name}' | {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        print("=========================================================\n")
        return {
            "status": "error",
            "message": error_msg,
            "query": query,
            "corpus_name": corpus_name,
        }
