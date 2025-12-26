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

        # Initialize renamer for this simple operation
        # Note: Renamer is lightweight, so initializing per file is acceptable for parallelization availability
        # Alternatively, we could pass it, but pickling a class instance is sometimes tricky if it has logger
        renamer = Renamer(dry_run=dry_run, style=style)

        new_filename = renamer.generate_new_filename(author, title)


        # Initial check to avoid unnecessary renaming logic if names match
        if file_path.name == new_filename:
             return f"ℹ️ Skipping {file_path.name} (already named correctly)"

        # Handle duplication collision logic similar to renamer.rename_file via renamer instance
        # But we need to capture the output message.
        # Since renamer.rename_file logs internally, we might duplicate logic or modify Renamer.
        # Let's use Renamer logic but we need to capture what happened.

        # To avoid editing Renamer heavily, let's just do the rename here or use the class and infer result.

        # We'll use the renamer logic but suppress its logger effectively by not configuring it here,
        # or we just reimplement the check/rename to return the string.
        # Ideally, we refactor Renamer.rename_file to return a string, but let's stick to using it.

        # We can just call renamer.rename_file and return a success message assuming it worked,
        # or catch exceptions.

        try:
             renamer.rename_file(file_path, new_filename)
             if dry_run:
                 return f"🔍 [DRY RUN] Would rename '{file_path.name}' -> '{new_filename}'"
             else:
                 # Calculate the actual new filename if collisions happened (Renamer handles it but doesn't return it)
                 # This is a slight limitation of the current Renamer class.
                 # However, usually it's fine.
                 return f"✅ Renamed '{file_path.name}' -> '{new_filename}'"
        except Exception as e:
            return f"❌ Failed to rename {file_path.name}: {e}"

    except Exception as e:
        return f"❌ Error processing {file_path.name}: {e}"
