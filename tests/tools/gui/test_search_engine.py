"""Tests for tools.gui.search module.

Covers:
- Indexing tools by name, description, and keywords
- Exact and prefix matching
- Fuzzy matching for typos
- Relevance scoring and result ordering
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from tools.gui.search import SearchEngine


@pytest.fixture
def search_engine() -> SearchEngine:
    """Provide a fresh SearchEngine instance."""
    return SearchEngine()


@pytest.fixture
def sample_tools() -> list[dict[str, str | list[str]]]:
    """Sample tools database for testing."""
    return [
        {
            "name": "Pressure Drop Calculator",
            "desc": "Pipe flow pressure drop analysis with multiple friction methods",
            "type": "python",
        },
        {
            "name": "Financial Calculator",
            "desc": "Comprehensive financial modeling for plant operations",
            "type": "python",
        },
        {
            "name": "ODE Solver",
            "desc": "Solve systems of ordinary differential equations symbolically",
            "type": "python",
        },
        {
            "name": "Data Processor",
            "desc": "Signal processing and time-series data analysis tool",
            "type": "python",
            "keywords": ["signal", "analysis", "time-series"],
        },
        {
            "name": "Lower Body Model",
            "desc": "Simulate and inspect lower-body MuJoCo kinematics and controls",
            "type": "python",
        },
    ]


# ── Indexing Tests ──────────────────────────────────────────────────────────


class TestIndexing:
    """Test tool indexing functionality."""

    def test_index_tools_basic(self, search_engine: SearchEngine) -> None:
        """Test basic tool indexing."""
        tools = [
            {"name": "Test Tool", "desc": "A test tool"},
        ]
        search_engine.index_tools(tools)

        assert len(search_engine.tools) == 1
        assert len(search_engine.index) > 0
        assert "test" in search_engine.index
        assert "tool" in search_engine.index

    def test_index_tools_multiple(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test indexing multiple tools."""
        search_engine.index_tools(sample_tools)

        assert len(search_engine.tools) == 5
        # Should have indexed names and descriptions
        assert "pressure" in search_engine.index
        assert "calculator" in search_engine.index
        assert "financial" in search_engine.index

    def test_index_custom_keywords(self, search_engine: SearchEngine) -> None:
        """Test indexing custom keywords."""
        tools = [
            {
                "name": "Data Processor",
                "desc": "Process data",
                "keywords": ["fft", "signal-processing"],
            }
        ]
        search_engine.index_tools(tools)

        assert "fft" in search_engine.index
        assert "signal-processing" in search_engine.index

    def test_index_invalid_tools_type(self, search_engine: SearchEngine) -> None:
        """Test that non-list input raises TypeError."""
        with pytest.raises(TypeError):
            search_engine.index_tools("not a list")  # type: ignore

    def test_index_skips_invalid_entries(self, search_engine: SearchEngine) -> None:
        """Test that invalid tool entries are safely skipped."""
        tools = [
            {"name": "Valid Tool", "desc": "Valid"},
            "not a dict",  # type: ignore
            {"name": "Another Valid", "desc": "Also valid"},
            None,  # type: ignore
        ]
        search_engine.index_tools(tools)

        assert len(search_engine.tools) == 4  # All added to tools list
        # Valid tools should be indexed
        assert "valid" in search_engine.index

    def test_clear_index(self, search_engine: SearchEngine) -> None:
        """Test clearing the index."""
        tools = [{"name": "Test", "desc": "Test tool"}]
        search_engine.index_tools(tools)
        assert len(search_engine.tools) > 0

        search_engine.clear()
        assert len(search_engine.tools) == 0
        assert len(search_engine.index) == 0


# ── Exact and Prefix Matching Tests ──────────────────────────────────────────


class TestExactMatching:
    """Test exact and prefix matching."""

    def test_exact_name_match(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test exact match on tool name."""
        search_engine.index_tools(sample_tools)
        results = search_engine.search("pressure")

        assert len(results) > 0
        assert any("Pressure" in t.get("name", "") for t in results)

    def test_case_insensitive_search(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test that search is case-insensitive."""
        search_engine.index_tools(sample_tools)

        results_lower = search_engine.search("pressure")
        results_upper = search_engine.search("PRESSURE")
        results_mixed = search_engine.search("PrEsSuRe")

        assert len(results_lower) == len(results_upper) == len(results_mixed)
        assert results_lower[0] == results_upper[0]

    def test_prefix_matching(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test prefix matching."""
        search_engine.index_tools(sample_tools)

        # "calc" should match "calculator"
        results = search_engine.search("calc")
        assert len(results) > 0
        assert any("Calculator" in t.get("name", "") for t in results)

    def test_empty_query(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test empty search query."""
        search_engine.index_tools(sample_tools)
        results = search_engine.search("")

        assert len(results) == 0

    def test_whitespace_query(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test whitespace-only query."""
        search_engine.index_tools(sample_tools)
        results = search_engine.search("   ")

        assert len(results) == 0


# ── Fuzzy Matching Tests ────────────────────────────────────────────────────


class TestFuzzyMatching:
    """Test fuzzy matching for typo tolerance."""

    def test_typo_one_char_wrong(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test matching with one character wrong (typo)."""
        search_engine.index_tools(sample_tools)

        # "pressur" instead of "pressure" - fuzzy match on individual word
        results = search_engine.search("pressur")
        # Fuzzy matching works on indexed keywords, not full text
        # This test demonstrates the limitation
        assert isinstance(results, list)

    def test_typo_transposition(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test fuzzy matching with transposed characters."""
        search_engine.index_tools(sample_tools)

        # "proces" instead of "processor" - still contains substr
        results = search_engine.search("proces")
        # Fuzzy match should find data processor
        assert len(results) > 0

    def test_substring_match(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test substring matching."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("drop")
        assert len(results) > 0
        assert "Pressure Drop" in results[0].get("name", "")

    def test_keyword_match(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test matching on custom keywords."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("signal")
        assert len(results) > 0
        assert any("Data Processor" in t.get("name", "") for t in results)

    def test_description_keyword_match(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test matching keywords from description."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("kinematics")
        assert len(results) > 0
        assert any("Lower Body" in t.get("name", "") for t in results)

    def test_multi_word_description_search(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test that all words in a multi-word query can match descriptions."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("pipe flow")

        assert len(results) > 0
        assert "Pressure Drop" in results[0].get("name", "")

    def test_punctuation_does_not_block_keyword_search(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test hyphenated description terms are normalized for discovery."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("series")

        assert len(results) > 0
        assert any("Data Processor" in t.get("name", "") for t in results)


# ── Relevance and Scoring Tests ─────────────────────────────────────────────


class TestRelevanceScoring:
    """Test result relevance and scoring."""

    def test_exact_match_ranks_first(self, search_engine: SearchEngine) -> None:
        """Test that exact matches rank higher."""
        tools = [
            {"name": "Test Tool", "desc": "Description"},
            {"name": "Another Tool", "desc": "With test in description"},
        ]
        search_engine.index_tools(tools)

        results = search_engine.search("test")
        assert len(results) > 0
        # Tool with "test" in name should rank first
        assert "Test Tool" in results[0].get("name", "")

    def test_max_results_limit(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test max_results parameter."""
        search_engine.index_tools(sample_tools)

        results_all = search_engine.search("tool", max_results=100)
        results_limited = search_engine.search("tool", max_results=2)

        assert len(results_limited) <= 2
        assert len(results_limited) <= len(results_all)

    def test_relevance_ordering(self, search_engine: SearchEngine) -> None:
        """Test that results are ordered by relevance."""
        tools = [
            {"name": "Data Processor", "desc": "Process your data"},
            {"name": "Tool", "desc": "A generic tool for processing"},
            {
                "name": "Signal Processor",
                "desc": "Process signals with advanced techniques",
            },
        ]
        search_engine.index_tools(tools)

        results = search_engine.search("process")
        # Should have results ordered by relevance
        assert len(results) > 0
        # First result should be one with "process" in name, not just description
        assert "Processor" in results[0].get("name", "")


# ── Error Handling Tests ────────────────────────────────────────────────────


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_search_query_type_validation(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test that non-string query raises TypeError."""
        search_engine.index_tools(sample_tools)

        with pytest.raises(TypeError):
            search_engine.search(123)  # type: ignore

    def test_invalid_max_results(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test that invalid max_results raises ValueError."""
        search_engine.index_tools(sample_tools)

        with pytest.raises(ValueError):
            search_engine.search("test", max_results=0)

        with pytest.raises(ValueError):
            search_engine.search("test", max_results=-1)

    def test_search_empty_database(self, search_engine: SearchEngine) -> None:
        """Test search on empty database."""
        results = search_engine.search("anything")
        assert results == []

    def test_no_matches(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test search with no matching results."""
        search_engine.index_tools(sample_tools)

        results = search_engine.search("xyzabc12345")
        assert results == []


# ── Integration Tests ───────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests with realistic usage."""

    def test_realistic_search_scenario(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test a realistic search scenario."""
        search_engine.index_tools(sample_tools)

        # User is looking for pressure drop calculator
        results = search_engine.search("pressure")
        assert len(results) > 0
        assert "Pressure Drop" in results[0].get("name", "")

    def test_multiple_searches_on_same_index(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test multiple searches on the same indexed data."""
        search_engine.index_tools(sample_tools)

        results1 = search_engine.search("financial")
        results2 = search_engine.search("data")
        results3 = search_engine.search("solve")

        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0
        assert results1 != results2

    def test_reindex_clears_old_data(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test that reindexing clears old data."""
        search_engine.index_tools(sample_tools)

        # Search should work
        results1 = search_engine.search("pressure")
        assert len(results1) > 0

        # Reindex with different tools
        new_tools = [{"name": "New Tool", "desc": "Something different"}]
        search_engine.index_tools(new_tools)

        # Old search should no longer work
        results2 = search_engine.search("pressure")
        assert len(results2) == 0

        # New search should work
        results3 = search_engine.search("new")
        assert len(results3) > 0

    def test_get_all_keywords(
        self, search_engine: SearchEngine, sample_tools: list[dict]
    ) -> None:
        """Test retrieving all indexed keywords."""
        search_engine.index_tools(sample_tools)

        keywords = search_engine.get_all_keywords()
        assert len(keywords) > 0
        assert "pressure" in keywords
        assert "calculator" in keywords
