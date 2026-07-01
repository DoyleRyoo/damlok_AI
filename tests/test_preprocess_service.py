import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.preprocess_service import (
    debug_tag_meeting_transcript,
    debug_preprocess_meeting_transcript,
    preprocess_meeting_transcripts,
    preprocess_meeting_transcript,
    preprocess_transcript_ai,
    tag_meeting_transcript,
    tag_transcript_ai,
)
from app.services.summary_service import SummaryServiceError


class PreprocessServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_basic_preprocess_mode_when_configured(self) -> None:
        with patch.dict("os.environ", {"PREPROCESS_MODE": "basic"}, clear=True):
            result = await preprocess_meeting_transcript("  첫 줄\t입니다.\r\n\r\n  둘째 줄  ")

        self.assertEqual(result, "첫 줄 입니다.\n\n둘째 줄")

    async def test_uses_light_ai_preprocess_mode_by_default(self) -> None:
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="첫 줄 입니다.\n\n둘째 줄"))]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "test-key",
                    "PREPROCESS_MODEL": "test-model",
                },
                clear=True,
            ),
            patch("app.services.preprocess_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.preprocess_service._chunk_text",
                return_value=["첫 줄 입니다.\n\n둘째 줄"],
            ),
        ):
            result = await preprocess_meeting_transcript("  첫 줄\t입니다.\r\n\r\n  둘째 줄  ")

        self.assertEqual(result, "첫 줄 입니다.\n\n둘째 줄")

    async def test_preprocess_response_separates_cleaned_and_readable_transcripts(
        self,
    ) -> None:
        result = await preprocess_meeting_transcripts("  원문\t입니다.  ")

        self.assertEqual(result.text, "원문 입니다.")
        self.assertEqual(result.raw_transcript, "  원문\t입니다.  ")
        self.assertEqual(result.cleaned_transcript, "원문 입니다.")
        self.assertIsNone(result.readable_transcript)
        self.assertFalse(result.readable_available)
        self.assertIsNone(result.readable_error)

    async def test_preprocess_response_does_not_call_ai_tagging(self) -> None:
        with patch(
            "app.services.preprocess_service.preprocess_transcript_ai",
            new=AsyncMock(side_effect=AssertionError("AI should not be called")),
        ):
            result = await preprocess_meeting_transcripts("  원문\t입니다.  ")

        self.assertEqual(result.text, "원문 입니다.")
        self.assertEqual(result.cleaned_transcript, "원문 입니다.")
        self.assertIsNone(result.readable_transcript)
        self.assertFalse(result.readable_available)
        self.assertIsNone(result.readable_error)

    async def test_tagging_returns_new_pipeline_tags(self) -> None:
        original = "서울청장은 현장 점검 계획을 수립해 주세요."
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content="ACTION: 서울청장은 현장 점검 계획을 수립해 주세요."
                )
            )
        ]
        create = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.completions.create = create

        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "test-key",
                    "TAGGING_MODEL": "tag-model",
                },
                clear=True,
            ),
            patch("app.services.preprocess_service.AsyncOpenAI", return_value=client),
            patch("app.services.preprocess_service._chunk_text", return_value=[original]),
        ):
            result = await tag_transcript_ai(original)

        self.assertEqual(result, "ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.")
        create.assert_awaited_once()
        self.assertEqual(create.await_args.kwargs["model"], "tag-model")
        self.assertIn("ACTION:", create.await_args.kwargs["messages"][0]["content"])
        self.assertIn("CHITCHAT:", create.await_args.kwargs["messages"][0]["content"])
        self.assertIn("ACTION은 넓게", create.await_args.kwargs["messages"][0]["content"])
        self.assertIn("DECISION은 좁게", create.await_args.kwargs["messages"][0]["content"])
        self.assertIn("INFO는 사고 개요", create.await_args.kwargs["messages"][0]["content"])

    async def test_tagging_response_includes_cleaned_and_tagged_transcripts(self) -> None:
        with patch(
            "app.services.preprocess_service.tag_transcript_ai",
            new=AsyncMock(return_value="DECISION: 검색 기능을 출시합니다."),
        ):
            result = await tag_meeting_transcript("  검색 기능을 출시합니다.  ")

        self.assertEqual(result.text, "DECISION: 검색 기능을 출시합니다.")
        self.assertEqual(result.raw_transcript, "  검색 기능을 출시합니다.  ")
        self.assertEqual(result.cleaned_transcript, "검색 기능을 출시합니다.")
        self.assertEqual(result.tagged_transcript, "DECISION: 검색 기능을 출시합니다.")

    async def test_ai_preprocess_tags_chunked_transcript(self) -> None:
        original = "서울청장은 안전 점검을 해 주시기 바랍니다."
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content="[액션후보] 서울청장은 안전 점검을 해 주시기 바랍니다."
                )
            )
        ]
        create = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.completions.create = create

        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "test-key",
                    "PREPROCESS_MODEL": "test-model",
                },
                clear=True,
            ),
            patch("app.services.preprocess_service.AsyncOpenAI", return_value=client),
            patch("app.services.preprocess_service._chunk_text", return_value=[original]),
        ):
            result = await preprocess_transcript_ai(original)

        self.assertEqual(result, "[액션후보] 서울청장은 안전 점검을 해 주시기 바랍니다.")
        create.assert_awaited_once()
        self.assertEqual(create.await_args.kwargs["model"], "test-model")
        self.assertIn(
            "섹션 태깅",
            create.await_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "[액션후보]",
            create.await_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "회의 STT 구간 1/1",
            create.await_args.kwargs["messages"][1]["content"],
        )

    async def test_tagging_debug_reports_pipeline_tag_counts(self) -> None:
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content="\n\n".join(
                        [
                            "ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.",
                            "DECISION: 안전 점검을 강화하기로 했습니다.",
                            "DISCUSSION: 지게차 사고 예방 방안을 논의했습니다.",
                            "CHITCHAT: 회의를 시작하겠습니다.",
                        ]
                    )
                )
            )
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        original = "서울청장은 현장 점검 계획을 수립해 주세요."

        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "test-key",
                    "TAGGING_MODEL": "test-model",
                },
                clear=True,
            ),
            patch("app.services.preprocess_service.AsyncOpenAI", return_value=client),
            patch("app.services.preprocess_service._chunk_text", return_value=[original]),
        ):
            result = await debug_tag_meeting_transcript(original)

        self.assertEqual(result.tag_counts["ACTION"], 1)
        self.assertEqual(result.tag_counts["DECISION"], 1)
        self.assertEqual(result.tag_counts["DISCUSSION"], 1)
        self.assertEqual(result.tag_counts["CHITCHAT"], 1)
        self.assertEqual(result.summary_used_tag_counts["ACTION"], 1)
        self.assertEqual(result.ignored_tag_counts["CHITCHAT"], 1)
        self.assertEqual(result.action_count, 1)
        self.assertEqual(result.decision_count, 1)
        self.assertEqual(result.discussion_count, 1)
        self.assertEqual(result.chitchat_count, 1)

    async def test_aggressive_ai_is_only_used_for_debug_view(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "PREPROCESS_MODE": "aggressive_ai",
                    "OPENAI_API_KEY": "test-key",
                },
                clear=True,
            ),
            patch(
                "app.services.preprocess_service.preprocess_transcript_ai",
                new=AsyncMock(return_value="분석용 light 결과"),
            ) as preprocess_ai,
        ):
            result = await preprocess_meeting_transcript("회의 전문")

        self.assertEqual(result, "분석용 light 결과")
        preprocess_ai.assert_awaited_once_with("회의 전문", "light_ai")

    async def test_ai_mode_requires_api_key(self) -> None:
        with patch.dict("os.environ", {"PREPROCESS_MODE": "ai"}, clear=True):
            with self.assertRaises(SummaryServiceError):
                await preprocess_meeting_transcript("회의 전문")

    async def test_debug_preprocess_returns_cleanup_only_lengths_and_previews(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "PREPROCESS_MODE": "ai",
                    "OPENAI_API_KEY": "test-key",
                    "PREPROCESS_DEBUG_PREVIEW_CHARS": "4",
                },
                clear=True,
            ),
            patch(
                "app.services.preprocess_service._preprocess_transcript_ai_result",
                new=AsyncMock(side_effect=AssertionError("AI should not be called")),
            ),
        ):
            result = await debug_preprocess_meeting_transcript("  어\t원문  ")

        self.assertEqual(result.mode, "basic")
        self.assertEqual(result.raw_transcript, "  어\t원문  ")
        self.assertEqual(result.cleaned_transcript, "어 원문")
        self.assertEqual(result.input_chars, 8)
        self.assertEqual(result.cleaned_chars, 4)
        self.assertEqual(result.removed_chars, 4)
        self.assertEqual(result.basic_chars, 4)
        self.assertIsNone(result.ai_chars)
        self.assertIsNone(result.raw_ai_chars)
        self.assertEqual(result.final_processed_chars, 4)
        self.assertEqual(result.total_chunks, 0)
        self.assertEqual(result.ai_used_chunks, 0)
        self.assertEqual(result.fallback_chunks, 0)
        self.assertEqual(result.too_short_count, 0)
        self.assertEqual(result.too_long_count, 0)
        self.assertTrue(result.preserved_dates)
        self.assertTrue(result.preserved_numbers)
        self.assertTrue(result.preserved_entities)
        self.assertTrue(result.preserved_directives)
        self.assertEqual(result.date_preserve_ratio, 1.0)
        self.assertEqual(result.number_preserve_ratio, 1.0)
        self.assertEqual(result.entity_preserve_ratio, 1.0)
        self.assertEqual(result.directive_preserve_ratio, 1.0)
        self.assertEqual(result.fallback_reason_by_chunk, [])
        self.assertFalse(result.fallback_used)
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.basic_preview, "어 원문")
        self.assertIsNone(result.ai_preview)

    async def test_rejects_unknown_preprocess_mode(self) -> None:
        with patch.dict("os.environ", {"PREPROCESS_MODE": "unknown"}):
            with self.assertRaises(SummaryServiceError):
                await preprocess_meeting_transcript("회의 전문")


if __name__ == "__main__":
    unittest.main()
