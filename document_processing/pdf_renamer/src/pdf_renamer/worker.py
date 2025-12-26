import logging
from pathlib import Path

from .extractor import extract_metadata
from .renamer import Renamer

logger = logging.getLogger(__name__)

def process_single_file(file_path: Path, style: str, dry_run: bool) -> str:
    """
    Worker function to process a single PDF file.
    Returns a status message.
    """
    try:
        if not file_path.exists():
            return f"❌ File not found: {file_path}"

        author, title = extract_metadata(file_path)

        if not author or not title:
            return f"⚠️ Skipping {file_path.name}: Missing metadata (Author: {author}, Title: {title})"

        # Initialize per file for parallelization (Renamer is lightweight and avoids pickling issues)
        renamer = Renamer(dry_run=dry_run, style=style)

        new_filename = renamer.generate_new_filename(author, title)


        # Initial check to avoid unnecessary renaming logic if names match
        if file_path.name == new_filename:
            return f"ℹ️ Skipping {file_path.name} (already named correctly)"

        # Use Renamer to perform the actual rename; it handles collisions internally and logs details.
        # This function then returns a concise status message based on the outcome.
        try:
            renamer.rename_file(file_path, new_filename)
            if dry_run:
                return f"🔍 [DRY RUN] Would rename '{file_path.name}' -> '{new_filename}'"
            else:
                # Note: Renamer may adjust the final filename on collision; we report the intended name here.
                return f"✅ Renamed '{file_path.name}' -> '{new_filename}'"
        except Exception as e:
            return f"❌ Failed to rename {file_path.name}: {e}"

    except Exception as e:
        return f"❌ Error processing {file_path.name}: {e}"
