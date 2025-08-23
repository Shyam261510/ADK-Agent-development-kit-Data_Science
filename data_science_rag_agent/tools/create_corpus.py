"""
Tool for creating a new Vertex AI RAG corpus.
"""

import re

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
)
from .utils import check_corpus_exists


def create_corpus(
    corpus_name: str,
    tool_context: ToolContext,
) -> dict:
    """
    Create a new Vertex AI RAG corpus with the specified name.

    Args:
        corpus_name (str): The name for the new corpus
        tool_context (ToolContext): The tool context for state management

    Returns:
        dict: Status information about the operation
    """
    print(f"[DEBUG] Starting create_corpus with corpus_name='{corpus_name}'")

    # Check if corpus already exists
    if check_corpus_exists(corpus_name, tool_context):
        print(f"[INFO] Corpus '{corpus_name}' already exists. Skipping creation.")
        return {
            "status": "info",
            "message": f"Corpus '{corpus_name}' already exists",
            "corpus_name": corpus_name,
            "corpus_created": False,
        }

    try:
        # Clean corpus name for use as display name
        display_name = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_name)
        print(
            f"[DEBUG] Cleaned display_name='{display_name}' from corpus_name='{corpus_name}'"
        )

        # Configure embedding model
        print(f"[DEBUG] Using embedding model '{DEFAULT_EMBEDDING_MODEL}'")
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=DEFAULT_EMBEDDING_MODEL
            )
        )

        # Create the corpus
        print("[DEBUG] Creating corpus with Vertex AI RAG...")
        rag_corpus = rag.create_corpus(
            display_name=display_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        print(
            f"[SUCCESS] Corpus created in Vertex AI: name='{rag_corpus.name}', display_name='{rag_corpus.display_name}'"
        )

        # Update state to track corpus existence
        tool_context.state[f"corpus_exists_{corpus_name}"] = True
        print(
            f"[DEBUG] Updated tool_context.state with corpus_exists_{corpus_name}=True"
        )

        # Set this as the current corpus
        tool_context.state["current_corpus"] = corpus_name
        print(f"[DEBUG] Set current_corpus='{corpus_name}' in tool_context.state")

        result = {
            "status": "success",
            "message": f"Successfully created corpus '{corpus_name}'",
            "corpus_name": rag_corpus.name,
            "display_name": rag_corpus.display_name,
            "corpus_created": True,
        }
        print(f"[RESULT] {result}")
        return result

    except Exception as e:
        print(f"[ERROR] Exception occurred while creating corpus: {str(e)}")
        return {
            "status": "error",
            "message": f"Error creating corpus: {str(e)}",
            "corpus_name": corpus_name,
            "corpus_created": False,
        }
