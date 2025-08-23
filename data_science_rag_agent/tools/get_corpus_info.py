"""
Tool for retrieving detailed information about a specific RAG corpus.
"""

from google.adk.tools.tool_context import ToolContext

from vertexai import rag

from .utils import check_corpus_exists, get_corpus_resource_name


def get_corpus_info(corpus_name: str, tool_context: ToolContext) -> dict:
    """
    Get detailed information about a specific RAG corpus including its files.

    Args:
         corpus_name (str): The full resoruce name of the corpus to get information about.
                            Prefreable use the resource_name form the list_corpora results
          tool_contex (ToolContext) : The tool context for state management

        Returns:
        Dict:Infomation about the corpus and its files
    """

    try:
        # check if corpus exists
        if not check_corpus_exists(corpus_name=corpus_name, tool_context=tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
            }

        # Get corpus resource name

        full_corpus_name = get_corpus_resource_name(corpus_name=corpus_name)

        corpus_display_name = corpus_name  # Default if we can't get actual display name

        # Process file information

        file_details = []

        try:
            # get the list of files in the corpus

            files = rag.list_files(full_corpus_name)

            for rag_file in files:
                # getting document specific details

                try:
                    # extracting the file id from the name
                    file_id = rag_file.name.split("/")[-1]

                    file_info = {
                        "file_id": file_id,
                        "diplay_name": (
                            rag_file.display_name
                            if hasattr(rag_file, "display_name")
                            else ""
                        ),
                        "resource_id": rag_file.google_drive_source.resource_ids[
                            0
                        ].resource_id,
                        "soruce_uri": (
                            rag_file.source_uri
                            if hasattr(rag_file, "source_uri")
                            else ""
                        ),
                        "create_time": (
                            rag_file.create_time
                            if hasattr(rag_file, "create_time")
                            else ""
                        ),
                        "upate_time": (
                            rag_file.update_time
                            if hasattr(rag_file, "update_time")
                            else ""
                        ),
                    }

                    file_details.append(file_info)

                except Exception:
                    # continue to the next file
                    continue

        except Exception as e:
            # continue without file details
            pass

            # Basic corpus info

        return {
            "status": "success",
            "message": f"Successfully retrieved information for corpus '{corpus_display_name}'",
            "corpus_name": corpus_name,
            "corpus_display_name": corpus_display_name,
            "file_count": len(file_details),
            "files": file_details,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": f"Error getting corpus information: {str(e)}",
            "corpus_name": corpus_name,
        }
