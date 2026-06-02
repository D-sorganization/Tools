"""Tests for the production peer-review LLM client (Tools #3177).

Covers:
- ``AdapterReviewerLLMClient.evaluate`` derives verdicts from a mocked
  adapter's ``send_message`` output (success path).
- Malformed / non-JSON adapter output degrades to ``abstain``.
- ``confidence`` is clamped into ``[0, 1]``.
- Adapter exceptions degrade to ``abstain`` instead of propagating.
- DbC: constructing with an adapter lacking ``send_message`` raises.
- ``registry.default_llm_client`` selects the production client when an
  adapter is available and the stub when offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.python.ai.peer_review._llm import (
    AdapterReviewerLLMClient,
    ReviewerLLMClient,
    StubReviewerLLMClient,
)
from shared.python.ai.peer_review.registry import default_llm_client

pytestmark = pytest.mark.unit


def _adapter_returning(content: str) -> MagicMock:
    """Build a mock adapter whose send_message returns the given content."""
    adapter = MagicMock()
    response = MagicMock()
    response.content = content
    adapter.send_message.return_value = response
    return adapter


class TestAdapterReviewerLLMClient:
    def test_is_reviewer_llm_client(self) -> None:
        """The production client satisfies the runtime-checkable protocol."""
        client = AdapterReviewerLLMClient(_adapter_returning("{}"))
        assert isinstance(client, ReviewerLLMClient)

    def test_rejects_adapter_without_send_message(self) -> None:
        """DbC: a bad adapter is a precondition violation (TypeError)."""
        bad = object()
        with pytest.raises(TypeError, match="send_message"):
            AdapterReviewerLLMClient(bad)  # type: ignore[arg-type]

    async def test_success_path_derives_verdict_from_adapter(self) -> None:
        adapter = _adapter_returning(
            '{"verdict": "request_changes", '
            '"reasoning": "missing tests", "confidence": 0.66}'
        )
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["correctness"],
            subject_content="some code",
            role="critic",
        )

        assert result["verdict"] == "request_changes"
        assert result["reasoning"] == "missing tests"
        assert result["confidence"] == pytest.approx(0.66)
        adapter.send_message.assert_called_once()

    async def test_prompt_includes_role_and_criteria(self) -> None:
        adapter = _adapter_returning('{"verdict": "approve", "confidence": 1.0}')
        client = AdapterReviewerLLMClient(adapter)

        await client.evaluate(
            criteria_set=["security", "performance"],
            subject_content="payload",
            role="advocate",
        )

        prompt = adapter.send_message.call_args.args[0]
        assert "advocate" in prompt
        assert "security" in prompt
        assert "performance" in prompt
        assert "payload" in prompt

    async def test_extracts_json_embedded_in_prose(self) -> None:
        """CLI agents wrap JSON in chatter / fences; we extract the object."""
        adapter = _adapter_returning(
            "Sure, here is my review:\n"
            '```json\n{"verdict": "approve", "reasoning": "ok", '
            '"confidence": 0.9}\n```\nHope that helps!'
        )
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "approve"
        assert result["confidence"] == pytest.approx(0.9)

    async def test_non_json_output_degrades_to_abstain(self) -> None:
        adapter = _adapter_returning("I cannot review this right now.")
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "abstain"
        assert result["confidence"] == 0.0

    async def test_malformed_json_degrades_to_abstain(self) -> None:
        adapter = _adapter_returning('{"verdict": "approve", "confidence":}')
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "abstain"

    async def test_unknown_verdict_degrades_to_abstain(self) -> None:
        adapter = _adapter_returning(
            '{"verdict": "looks_good_to_me", "confidence": 0.8}'
        )
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "abstain"

    async def test_confidence_clamped_above_one(self) -> None:
        adapter = _adapter_returning('{"verdict": "approve", "confidence": 4.2}')
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "approve"
        assert result["confidence"] == 1.0

    async def test_confidence_clamped_below_zero(self) -> None:
        adapter = _adapter_returning('{"verdict": "reject", "confidence": -3.0}')
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "reject"
        assert result["confidence"] == 0.0

    async def test_non_numeric_confidence_degrades_to_zero(self) -> None:
        adapter = _adapter_returning('{"verdict": "approve", "confidence": "high"}')
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "approve"
        assert result["confidence"] == 0.0

    async def test_adapter_exception_degrades_to_abstain(self) -> None:
        adapter = MagicMock()
        adapter.send_message.side_effect = RuntimeError("provider exploded")
        client = AdapterReviewerLLMClient(adapter)

        result = await client.evaluate(
            criteria_set=["x"], subject_content="y", role="critic"
        )
        assert result["verdict"] == "abstain"
        assert "provider exploded" in str(result["reasoning"])


class TestDefaultLlmClient:
    def test_selects_production_client_when_adapter_available(self) -> None:
        mock_adapter = MagicMock()
        mock_adapter.send_message = MagicMock()

        with patch(
            "src.shared.python.ai.adapters.factory.AdapterFactory.get_best_available",
            return_value=mock_adapter,
        ):
            client = default_llm_client()

        assert isinstance(client, AdapterReviewerLLMClient)

    def test_falls_back_to_stub_when_offline(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.factory.AdapterFactory.get_best_available",
            return_value=None,
        ):
            client = default_llm_client()

        assert isinstance(client, StubReviewerLLMClient)

    def test_falls_back_to_stub_on_factory_error(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.factory.AdapterFactory.get_best_available",
            side_effect=RuntimeError("no providers configured"),
        ):
            client = default_llm_client()

        assert isinstance(client, StubReviewerLLMClient)
