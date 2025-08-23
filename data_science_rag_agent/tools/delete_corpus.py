"""
Tool for deleting a Vertex AI RAG corpus when it's no longer needed.
"""

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from .utils import check_corpus_exists, get_corpus_resource_name


def delete_corpus(corpus_name: str, confirm: bool, tool_context: ToolContext) -> dict:
    """
    Delete a Vertex AI RAG corpus when it's no longer needed.
    Requires confirmation to prevent accidental deletion.


    Args:

        corpus_name(str):  The full resource name of the corpus to delete.
        confirm (bool): Must be set to True to confirm deletion
        tool_context (ToolContext) : The tool context for state management

    Returns:
    dict: Status information about the deletion operation
    """

    # Check if corpus exists

    if not check_corpus_exists(corpus_name=corpus_name, tool_context=tool_context):
        return {
            "status": "error",
            "message": f"Corpus '{corpus_name}' does not exist",
            "corpus_name": corpus_name,
        }

    # cheeck if deletion is confirmed

    if not confirm:
        return {
            "status": "error",
            "message": "Deletion requires explicit confirmation. Set confirm=True to delete this corpus.",
            "corpus_name": corpus_name,
        }

    try:
        # Get the corpus resoruce name
        full_corpus_name = get_corpus_resource_name(corpus_name=corpus_name)

        # Delete the corpus
        rag.delete_corpus(full_corpus_name)

        # Remove from state by setting to false

        state_key = f"corpus_exists_{corpus_name}"

        if state_key in tool_context.state:
            tool_context.state[state_key] = False

        return {
            "status": "success",
            "message": f"Successfully deleted corpus '{corpus_name}'",
            "corpus_name": corpus_name,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting corpus: {str(e)}",
            "corpus_name": corpus_name,
        }
