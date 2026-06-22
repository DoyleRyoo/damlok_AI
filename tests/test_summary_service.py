import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas import (
    ActionItemPriority,
    ActionItemStatus,
    AnalyzeResponse,
    ChunkAnalysis,
)
from app.services.summary_service import (
    SummaryServiceError,
    _chunk_text,
    _parse_analyze_result,
    analyze_meeting,
)


VALID_RESULT = {
    "summary": {
        "objective": "신규 기능의 출시 범위를 정한다.",
        "discussion": "필수 기능과 일정 위험을 논의했다.",
        "decision": "검색 기능을 우선 출시하기로 했다.",
    },
    "action_items": [
        {
            "assignee_name": "김담당",
            "assignee_email": None,
            "task": "검색 API를 구현한다.",
            "start_date": None,
            "due_date": "2026-06-30",
            "priority": "HIGH",
            "status": "미착수",
        }
    ],
    "meeting_summary": "검색 기능 우선 출시와 담당 업무를 결정한 회의",
}


class ParseAnalyzeResultTest(unittest.TestCase):
    def test_parses_valid_result(self) -> None:
        result = _parse_analyze_result(json.dumps(VALID_RESULT, ensure_ascii=False))

        self.assertEqual(result.summary.decision, "검색 기능을 우선 출시하기로 했다.")
        self.assertIs(result.action_items[0].priority, ActionItemPriority.HIGH)
        self.assertIs(result.action_items[0].status, ActionItemStatus.NOT_STARTED)

    def test_rejects_invalid_result(self) -> None:
        for content in (None, "", "not-json", "{}"):
            with self.subTest(content=content):
                with self.assertRaises(SummaryServiceError):
                    _parse_analyze_result(content)


class ChunkTextTest(unittest.TestCase):
    def test_splits_with_overlap_without_duplicate_tail(self) -> None:
        encoding = MagicMock()
        encoding.encode.side_effect = lambda text: list(text)
        encoding.decode.side_effect = lambda tokens: "".join(tokens)

        with (
            patch.dict(
                os.environ,
                {"SUMMARY_CHUNK_TOKENS": "5", "SUMMARY_CHUNK_OVERLAP": "1"},
            ),
            patch(
                "app.services.summary_service.tiktoken.encoding_for_model",
                return_value=encoding,
            ),
        ):
            chunks = _chunk_text("abcdefghij", "test-model")

        self.assertEqual(chunks, ["abcde", "efghi", "ij"])

    def test_does_not_create_overlap_only_chunk(self) -> None:
        encoding = MagicMock()
        encoding.encode.side_effect = lambda text: list(text)
        encoding.decode.side_effect = lambda tokens: "".join(tokens)

        with (
            patch.dict(
                os.environ,
                {"SUMMARY_CHUNK_TOKENS": "5", "SUMMARY_CHUNK_OVERLAP": "1"},
            ),
            patch(
                "app.services.summary_service.tiktoken.encoding_for_model",
                return_value=encoding,
            ),
        ):
            chunks = _chunk_text("abcde", "test-model")

        self.assertEqual(chunks, ["abcde"])


class AnalyzeMeetingTest(unittest.IsolatedAsyncioTestCase):
    async def test_analyzes_chunks_then_merges_once(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
            }
        )
        final_result = AnalyzeResponse.model_validate(VALID_RESULT)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        parse = AsyncMock(
            side_effect=[chunk_response, chunk_response, final_response]
        )
        client = MagicMock()
        client.chat.completions.parse = parse

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.summary_service._chunk_text",
                return_value=["첫 번째 구간", "두 번째 구간"],
            ),
        ):
            result = await analyze_meeting("회의 전문")

        self.assertEqual(result.meeting_summary, VALID_RESULT["meeting_summary"])
        self.assertEqual(parse.await_count, 3)
        first_chunk, second_chunk, merge_request = [
            call.kwargs for call in parse.await_args_list
        ]
        self.assertIn("회의록 구간 1/2", first_chunk["messages"][1]["content"])
        self.assertIn("회의록 구간 2/2", second_chunk["messages"][1]["content"])
        self.assertIs(first_chunk["response_format"], ChunkAnalysis)
        self.assertIs(second_chunk["response_format"], ChunkAnalysis)
        self.assertIs(merge_request["response_format"], AnalyzeResponse)
        self.assertIn("검색 API를 구현한다.", merge_request["messages"][1]["content"])

    async def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SummaryServiceError):
                await analyze_meeting("회의 전문")


if __name__ == "__main__":
    unittest.main()
