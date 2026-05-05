"""Search engine for tool discovery in the Unified Launcher.

Provides fuzzy matching and keyword-based search functionality to help
users find tools quickly without knowing exact names.
"""

import re
from typing import Any

_IGNORED_TERMS = frozenset({"and", "the", "for", "with"})
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class SearchEngine:
    """Indexes and searches tools using fuzzy matching and keyword scoring.

    Supports:
    - Name matching (exact and fuzzy)
    - Description keyword search
    - Custom keyword matching
    - Relevance scoring
    """

    def __init__(self) -> None:
        """Initialize the search engine."""
        self.tools: list[dict[str, Any]] = []
        self.index: dict[str, list[int]] = {}  # keyword -> list of tool indices

    def index_tools(self, tools: list[Any]) -> None:
        """Build search index from tools list.

        Args:
            tools: List of tool dictionaries with 'name', 'desc' keys.
        """
        if not isinstance(tools, list):
            raise TypeError("tools must be a list")

        self.tools = tools
        self.index.clear()

        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue

            # Index name.
            name = tool.get("name", "")
            if isinstance(name, str):
                for word in self._tokenize(name):
                    self._add_to_index(word, idx)

            # Index description keywords.
            desc = tool.get("desc", "")
            if isinstance(desc, str):
                for word in self._tokenize(desc):
                    if len(word) > 2 and word not in _IGNORED_TERMS:
                        self._add_to_index(word, idx)

            # Index custom keywords if provided
            keywords = tool.get("keywords", [])
            if isinstance(keywords, list):
                for keyword in keywords:
                    if isinstance(keyword, str):
                        self._add_to_index(keyword.lower(), idx)
                        for word in self._tokenize(keyword):
                            self._add_to_index(word, idx)

    def _add_to_index(self, keyword: str, tool_idx: int) -> None:
        """Add a keyword -> tool mapping to the index."""
        if keyword not in self.index:
            self.index[keyword] = []
        if tool_idx not in self.index[keyword]:
            self.index[keyword].append(tool_idx)

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search for tools matching the query.

        Uses fuzzy matching for typo tolerance and scores results by relevance.

        Args:
            query: Search query (name, description, or keywords).
            max_results: Maximum number of results to return.

        Returns:
            Sorted list of matching tools, best matches first.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")

        if not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_tools: dict[int, float] = {}
        for query_token in query_tokens:
            token_scores = self._score_token(query_token)
            if not token_scores:
                return []

            if not scored_tools:
                scored_tools = token_scores
                continue

            scored_tools = {
                idx: score + token_scores[idx]
                for idx, score in scored_tools.items()
                if idx in token_scores
            }

            if not scored_tools:
                return []

        # Sort by score (descending) and return
        sorted_results = sorted(scored_tools.items(), key=lambda x: x[1], reverse=True)
        return [self.tools[idx] for idx, _score in sorted_results[:max_results]]

    def _score_token(self, query_token: str) -> dict[int, float]:
        """Return matching tool indexes and relevance scores for one token."""
        scored_tools: dict[int, float] = {}

        # 1. Exact and prefix matches (highest score).
        for keyword, tool_indices in self.index.items():
            if keyword.startswith(query_token):
                for idx in tool_indices:
                    score = 1.0 if keyword == query_token else 0.8
                    scored_tools[idx] = max(scored_tools.get(idx, 0), score)

        # 2. Fuzzy matching on indexed keywords for typos.
        for keyword, tool_indices in self.index.items():
            if self._fuzzy_match_word(query_token, keyword):
                for idx in tool_indices:
                    scored_tools[idx] = max(scored_tools.get(idx, 0), 0.7)

        # 3. Partial word matches.
        for keyword, tool_indices in self.index.items():
            if query_token in keyword:
                for idx in tool_indices:
                    scored_tools[idx] = max(scored_tools.get(idx, 0), 0.5)

        return scored_tools

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        """Normalize free text into lowercase searchable tokens."""
        return _TOKEN_PATTERN.findall(value.lower())

    def _fuzzy_match_word(self, query: str, word: str, max_distance: int = 2) -> bool:
        """Check if a word is similar to query within edit distance.

        Args:
            query: Search term.
            word: Indexed keyword to match against.
            max_distance: Maximum edit distance allowed.

        Returns:
            True if word is within max_distance edits of query.
        """
        if not query or not word:
            return False

        # Levenshtein distance calculation for individual words
        distance = self._levenshtein_distance(query, word)
        return distance <= max_distance

    def _fuzzy_match(self, query: str, text: str, max_distance: int = 2) -> bool:
        """Check if text matches query with up to max_distance edits.

        Uses Levenshtein distance for basic fuzzy matching.

        Args:
            query: Search term.
            text: Text to match against.
            max_distance: Maximum edit distance allowed.

        Returns:
            True if text is within max_distance edits of query.
        """
        if not query or not text:
            return False

        # Check if query is a substring (already a match)
        if query in text:
            return True

        # Levenshtein distance calculation
        distance = self._levenshtein_distance(query, text)
        return distance <= max_distance

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance (minimum number of single-character edits).
        """
        if len(s1) < len(s2):
            return SearchEngine._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def clear(self) -> None:
        """Clear all indexed tools."""
        self.tools.clear()
        self.index.clear()

    def get_all_keywords(self) -> list[str]:
        """Return list of all indexed keywords."""
        return sorted(self.index.keys())
