"""API-only mode for PDF renaming - manual review and approval workflow."""

import logging
from pathlib import Path

from .cache import ResultCache
from .extractors import TitleLLM, author_from_metadata
from .types import TitleResult
from .utils import (
    get_last_name,
    sanitize_filename,
    sha256_file,
    to_kebab_case,
    to_snake_case,
    to_title_case,
)

logger = logging.getLogger(__name__)


class RenameProposal:
    """A proposed rename operation for manual review."""

    def __init__(
        self,
        file_path: Path,
        current_name: str,
        proposed_name: str,
        title_result: TitleResult,
        author: str = "",
        confidence: float = 0.0,
    ):
        self.file_path = file_path
        self.current_name = current_name
        self.proposed_name = proposed_name
        self.title_result = title_result
        self.author = author
        self.confidence = confidence
        self.approved = False
        self.rejected = False
        self.custom_name: str | None = None


class APIRenameManager:
    """Manages API-only rename operations with manual approval workflow."""

    def __init__(
        self,
        directory: Path,
        cache: ResultCache,
        llm: TitleLLM,
        style: str = "standard",
        include_author: bool = False,
        recursive: bool = True,
    ):
        self.directory = directory
        self.cache = cache
        self.llm = llm
        self.style = style
        self.include_author = include_author
        self.recursive = recursive
        self.proposals: list[RenameProposal] = []

    def generate_proposals(self) -> list[RenameProposal]:
        """Generate rename proposals using API calls only."""
        logger.info(f"Generating API-based rename proposals for: {self.directory}")

        # Find PDF files
        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        pdf_files = list(self.directory.glob(pattern))
        pdf_files = [f for f in pdf_files if f.is_file() and not f.is_symlink()]

        self.proposals = []

        for pdf_file in pdf_files:
            try:
                proposal = self._create_proposal(pdf_file)
                if proposal:
                    self.proposals.append(proposal)
                    logger.info(
                        f"Generated proposal: {proposal.current_name} -> {proposal.proposed_name}"
                    )
                else:
                    logger.warning(f"Could not generate proposal for: {pdf_file.name}")

            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Error processing {pdf_file}: {e}")

        logger.info(f"Generated {len(self.proposals)} rename proposals")
        return self.proposals

    def _create_proposal(self, file_path: Path) -> RenameProposal | None:
        """Create a single rename proposal."""
        try:
            # Calculate hash and check cache
            file_hash = sha256_file(file_path)
            cached = self.cache.get(file_hash)

            if cached and cached.title:
                result = cached
                logger.debug(f"[CACHE] {file_path.name} -> {result.title}")
            else:
                # Force API extraction (no local fallback)
                if not self.llm:
                    logger.warning(
                        f"No LLM available for API-only mode: {file_path.name}"
                    )
                    return None

                result = self.llm.extract_title(file_path)

                # Save to cache
                model_name = getattr(self.llm, "DEFAULT_MODEL", "unknown")
                self.cache.save(
                    file_hash,
                    file_path,
                    result,
                    provider="gemini",
                    model=model_name,
                )
                logger.info(
                    f"[API] {file_path.name} -> {result.title} ({result.confidence:.2f})"
                )

            if not result.title:
                logger.warning(f"No title extracted for: {file_path.name}")
                return None

            # Extract author if needed
            author = ""
            if self.include_author:
                author = author_from_metadata(file_path) or ""

            # Generate proposed filename
            proposed_name = self._generate_filename(result.title, author)

            return RenameProposal(
                file_path=file_path,
                current_name=file_path.name,
                proposed_name=proposed_name,
                title_result=result,
                author=author,
                confidence=result.confidence,
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error creating proposal for {file_path}: {e}")
            return None

    def _generate_filename(self, title: str, author: str) -> str:
        """Generate filename based on title, author, and style."""
        if self.style == "snake_case":
            clean_title = to_snake_case(title)
            if self.include_author and author:
                clean_author = to_snake_case(get_last_name(author))
                return f"{clean_author}_{clean_title}.pdf"
            return f"{clean_title}.pdf"

        elif self.style == "kebab_case":
            clean_title = to_kebab_case(title)
            if self.include_author and author:
                clean_author = to_kebab_case(get_last_name(author))
                return f"{clean_author}-{clean_title}.pdf"
            return f"{clean_title}.pdf"

        else:  # standard
            clean_title = sanitize_filename(to_title_case(title))
            if self.include_author and author:
                clean_author = sanitize_filename(get_last_name(author))
                return f"{clean_author} - {clean_title}.pdf"
            return f"{clean_title}.pdf"

    def approve_proposal(self, index: int, custom_name: str | None = None) -> bool:
        """Approve a rename proposal, optionally with a custom name."""
        if 0 <= index < len(self.proposals):
            proposal = self.proposals[index]
            proposal.approved = True
            proposal.rejected = False
            if custom_name:
                proposal.custom_name = custom_name
            logger.info(
                f"Approved: {proposal.current_name} -> {custom_name or proposal.proposed_name}"
            )
            return True
        return False

    def reject_proposal(self, index: int) -> bool:
        """Reject a rename proposal."""
        if 0 <= index < len(self.proposals):
            proposal = self.proposals[index]
            proposal.approved = False
            proposal.rejected = True
            logger.info(f"Rejected: {proposal.current_name}")
            return True
        return False

    def get_approved_proposals(self) -> list[RenameProposal]:
        """Get all approved proposals."""
        return [p for p in self.proposals if p.approved and not p.rejected]

    def get_pending_proposals(self) -> list[RenameProposal]:
        """Get all pending (not approved or rejected) proposals."""
        return [p for p in self.proposals if not p.approved and not p.rejected]

    def execute_approved_renames(self, dry_run: bool = True) -> dict[str, int]:
        """Execute all approved rename operations."""
        approved = self.get_approved_proposals()
        results = {"success": 0, "failed": 0, "skipped": 0}

        logger.info(f"Executing {len(approved)} approved renames (dry_run={dry_run})")

        for proposal in approved:
            try:
                final_name = proposal.custom_name or proposal.proposed_name
                target_path = proposal.file_path.parent / final_name

                # Check if already correctly named
                if target_path == proposal.file_path:
                    logger.info(f"Already correctly named: {proposal.current_name}")
                    results["skipped"] += 1
                    continue

                # Handle collisions
                if target_path.exists() and target_path != proposal.file_path:
                    file_hash = sha256_file(proposal.file_path)
                    short_hash = file_hash[:6]
                    stem = target_path.stem
                    target_path = proposal.file_path.parent / f"{stem}_{short_hash}.pdf"

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would rename: {proposal.current_name} -> {target_path.name}"
                    )
                    results["success"] += 1
                else:
                    proposal.file_path.rename(target_path)
                    logger.info(
                        f"Renamed: {proposal.current_name} -> {target_path.name}"
                    )
                    results["success"] += 1

            except (PermissionError, OSError) as e:
                logger.error(f"Failed to rename {proposal.current_name}: {e}")
                results["failed"] += 1

        return results

    def export_proposals_csv(self, output_path: Path) -> None:
        """Export proposals to CSV for external review."""
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Current Name",
                    "Proposed Name",
                    "Title",
                    "Author",
                    "Confidence",
                    "Method",
                    "Approved",
                    "Rejected",
                    "Custom Name",
                ]
            )

            for proposal in self.proposals:
                writer.writerow(
                    [
                        proposal.current_name,
                        proposal.proposed_name,
                        proposal.title_result.title,
                        proposal.author,
                        f"{proposal.confidence:.2f}",
                        proposal.title_result.method,
                        proposal.approved,
                        proposal.rejected,
                        proposal.custom_name or "",
                    ]
                )

        logger.info(f"Exported {len(self.proposals)} proposals to: {output_path}")
