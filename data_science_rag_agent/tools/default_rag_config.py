import re
from .create_corpus import create_corpus
from .get_corpus_info import get_corpus_info
from .add_data import add_data
from .utils import check_corpus_exists, get_corpus_resource_name
from google.adk.tools.tool_context import ToolContext


def default_rag_config(tool_context: ToolContext) -> dict:
    """
    Configure and set up a default Vertex AI RAG resource.

    This function checks whether a specific RAG corpus already exists.
    If not, it creates a new corpus, adds a given document, and ensures
    no duplicate entries are added.

    Args:
        tool_context (ToolContext): Context object for state management.

    Returns:
        dict: A status dictionary with success flag and message.
    """
    try:
        data_science_corpus = "data_science_agent"
        document_url = (
            "https://drive.google.com/file/d/1jN5t9ldRyDgExvzkEtIUnhMHLkTEynrr/view"
        )

        print("\n========== ⚙️ Starting Default RAG Configuration ==========")

        # --- Validate Google Drive URL ---
        print("🔍 Validating provided document URL...")
        drive_match = re.match(
            r"https:\/\/drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)(?:\/|$)",
            document_url,
        )
        if drive_match:
            file_id = drive_match.group(1)
            print(f"✅ Valid Google Drive file detected | File ID: {file_id}")
        else:
            print(f"❌ Invalid Google Drive URL format: {document_url}")
            return {"success": False, "message": "Invalid Google Drive URL format."}

        # --- Get corpus resource name ---
        full_corpus_name = get_corpus_resource_name(corpus_name=data_science_corpus)
        print(f"📌 Corpus Resource Name: {full_corpus_name}")

        # --- Check if corpus already exists ---
        print("🔍 Checking if corpus already exists...")
        exists = check_corpus_exists(
            corpus_name=full_corpus_name, tool_context=tool_context
        )
        print(f"➡️ Corpus Exists: {'Yes' if exists else 'No'}")

        corpus_info = get_corpus_info(
            corpus_name=full_corpus_name, tool_context=tool_context
        )
        # print(f"📊 Corpus Info: {corpus_info}")

        # --- Check for existing file in corpus ---
        print("🔍 Checking if the file already exists in the corpus...")
        is_file_exists = any(
            file_id == file.get("resource_id") for file in corpus_info.get("files", [])
        )

        if is_file_exists:
            print(f"⚠️ File with ID '{file_id}' already exists in the corpus.")
            return {
                "success": True,
                "message": f"File with ID '{file_id}' already exists in the corpus.",
                "corpus_name": full_corpus_name,
            }

        # --- Create new corpus if not found ---
        print("🆕 Creating new corpus...")
        new_corpus = create_corpus(
            corpus_name=data_science_corpus, tool_context=tool_context
        )
        print(f"✅ New Corpus Created: {new_corpus}")

        # --- Add document to corpus ---
        full_corpus_name = new_corpus.get("corpus_name")
        print(f"📥 Adding document '{document_url}' to corpus '{full_corpus_name}'...")
        new_data = add_data(
            corpus_name=full_corpus_name,
            paths=[document_url],
            tool_context=tool_context,
        )
        print(f"✅ New Data Added: {new_data}")

        print("🎉 Setup Complete: New corpus and document added successfully.")
        print("===========================================================\n")

        return {
            "success": True,
            "message": "New corpus and document added successfully.",
            "corpus_name": full_corpus_name,
        }

    except Exception as e:
        print("❌ ERROR during RAG setup:", str(e))
        return {"success": False, "message": f"Error while setting up RAG: {str(e)}"}
