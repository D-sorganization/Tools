import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import ResultCache
from .core import extract_title
from .llm_layer import GeminiTitleLLM
from .types import TitleResult
from .utils import sanitize_filename, sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("pdf_renamer")


def apply_style(title: str, style: str) -> str:
    safe_title = sanitize_filename(title)
    if style == "snake_case":
        return safe_title.lower().replace(" ", "_")
    elif style == "kebab_case":
        return safe_title.lower().replace(" ", "-")
    return safe_title


def process_file(
    file_path: Path,
    cache: ResultCache,
    llm: GeminiTitleLLM | None,
    dry_run: bool,
    style: str,
) -> None:
    try:
        # 1. Hash
        file_hash = sha256_file(file_path)

        # 2. Check Cache
        cached = cache.get(file_hash)
        result: TitleResult
        if cached and cached.title:
            result = cached
            logger.info(f"[CACHE] {file_path.name} -> {result.title}")
        else:
            # 3. Extract
            result = extract_title(file_path, llm)
            # 4. Save to Cache
            model_name = getattr(llm, "DEFAULT_MODEL", "unknown") if llm else "local"
            cache.save(
                file_hash,
                file_path,
                result,
                provider="gemini" if llm else "local",
                model=model_name if llm else "heuristic",
            )
            logger.info(
                f"[{result.method.upper()}] {file_path.name} -> {result.title} "
                f"({result.confidence:.2f})"
            )

        # 5. Rename if we have a title
        if result.title:
            new_stem = apply_style(result.title, style)
            new_name = f"{new_stem}.pdf"
            target = file_path.parent / new_name

            if target != file_path:
                if not dry_run:
                    if target.exists():
                        # Simple collision handling: append hash snippet
                        short_hash = file_hash[:6]
                        target = file_path.parent / f"{new_stem}_{short_hash}.pdf"

                    if target.exists():
                        logger.warning(f"Target still exists, skipping: {target}")
                    else:
                        file_path.rename(target)
                        logger.info(f"Renamed: {new_name}")
                else:
                    logger.info(f"[DRY RUN] Would rename to: {new_name}")

    except (IOError, PermissionError, OSError) as e:
        logger.error(f"Failed to process {file_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced PDF Renamer with AI Fallback"
    )
    parser.add_argument("directory", type=Path, help="Target directory")
    parser.add_argument("--db", type=Path, default=Path("pdf_titles.sqlite"))
    parser.add_argument(
        "--provider",
        choices=["gemini"],
        default="gemini",
        help="LLM provider for fallback",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without renaming"
    )
    parser.add_argument(
        "--style",
        choices=["standard", "snake_case", "kebab_case"],
        default="standard",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()

    target_dir: Path = args.directory
    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        sys.exit(1)

    cache = ResultCache(args.db)

    llm = None
    if args.provider == "gemini":
        llm = GeminiTitleLLM()

    files = list(target_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(files)} PDFs in {target_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        # Utilize map to ensure order and cleaner execution.
        from functools import partial

        process_func = partial(
            process_file, cache=cache, llm=llm, dry_run=args.dry_run, style=args.style
        )
        # Force execution
        list(exe.map(process_func, files))


if __name__ == "__main__":
    main()
