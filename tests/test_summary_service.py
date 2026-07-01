import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas import (
    AiReviewActionItemData,
    ActionItemConfidence,
    ActionItemPriority,
    ActionItemSourceType,
    ActionItemStatus,
    AnalyzeResponse,
    ChunkAnalysis,
)
from app.services.summary_service import (
    CHUNK_SYSTEM_PROMPT,
    MERGE_SYSTEM_PROMPT,
    SummaryServiceError,
    analyze_meeting_debug,
    _chunk_text,
    _parse_analyze_result,
    analyze_meeting,
    select_summary_sections,
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
            "source_type": "DIRECTIVE",
            "confidence": "HIGH",
            "evidence": "김담당은 검색 API를 구현해 주세요.",
            "duplicate_group_id": None,
        }
    ],
    "action_candidates": [
        {
            "assignee_name": None,
            "assignee_email": None,
            "task": "관리자 권한 정책을 검토한다.",
            "start_date": None,
            "due_date": None,
            "priority": "MEDIUM",
            "status": "미착수",
            "source_type": "SUGGESTION",
            "confidence": "MEDIUM",
            "evidence": "관리자 권한 정책은 추가 검토가 필요합니다.",
            "duplicate_group_id": "dup-1",
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
        self.assertIs(result.action_items[0].source_type, ActionItemSourceType.DIRECTIVE)
        self.assertIs(result.action_items[0].confidence, ActionItemConfidence.HIGH)
        self.assertEqual(result.action_items[0].evidence, "김담당은 검색 API를 구현해 주세요.")
        self.assertEqual(result.action_candidates[0].duplicate_group_id, "dup-1")

    def test_converts_null_like_action_item_values_to_none(self) -> None:
        payload = {
            **VALID_RESULT,
            "action_items": [
                {
                    **VALID_RESULT["action_items"][0],
                    "assignee_name": "확인되지 않음",
                    "assignee_email": "null",
                    "start_date": "null",
                    "due_date": "미정",
                }
            ],
        }

        result = _parse_analyze_result(json.dumps(payload, ensure_ascii=False))

        self.assertIsNone(result.action_items[0].assignee_name)
        self.assertIsNone(result.action_items[0].assignee_email)
        self.assertIsNone(result.action_items[0].start_date)
        self.assertIsNone(result.action_items[0].due_date)

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


class SelectSummarySectionsTest(unittest.TestCase):
    def test_selects_priority_tagged_sections(self) -> None:
        text = "\n\n".join(
            [
                "[의전/잡담] 인사말입니다.",
                "[핵심논의] 안전 점검 기준을 논의했습니다.",
                "[결정사항] 다음 회의에서 확정하기로 했습니다.",
                "[액션후보] 서울청장은 현장 점검을 해 주시기 바랍니다.",
                "[응답] 알겠습니다.",
            ]
        )

        selected = select_summary_sections(text)

        self.assertIn("[핵심논의]", selected)
        self.assertIn("[결정사항]", selected)
        self.assertIn("[액션후보]", selected)
        self.assertNotIn("[의전/잡담]", selected)
        self.assertNotIn("[응답]", selected)

    def test_selected_sections_make_debug_checks_observable(self) -> None:
        text = "\n\n".join(
            [
                "[의전/잡담] 잡담표식-ZZZ 오늘 날씨 이야기를 했습니다.",
                "[결정사항] 결정-1 검색 기능은 MVP에 포함합니다.",
                "[결정사항] 결정-2 관리자 권한은 다음 배포로 미룹니다.",
                "[결정사항] 결정-3 QA 완료 후 배포합니다.",
                "[결정사항] 결정-4 모바일 화면은 현 디자인을 유지합니다.",
                "[결정사항] 결정-5 로그 수집 범위를 축소합니다.",
                "[결정사항] 결정-6 알림 기능은 베타에서 제외합니다.",
                "[액션후보] 액션-1 민수님은 검색 API QA 체크리스트를 작성해 주세요.",
                "[액션후보] 액션-2 담당자는 추후 정하겠습니다. QA 문서를 준비해야 합니다.",
                "[액션후보] 액션-3 지영님은 2026-07-10까지 권한 검토 결과를 공유해 주세요.",
                "[액션후보] 액션-4 이번 주 금요일까지 배포하면 좋겠습니다.",
                "[액션후보] 액션-5 수진님은 로그 수집 정책을 정리해 주세요.",
                "[응답] 알겠습니다.",
            ]
        )

        selected = select_summary_sections(text)

        for index in range(1, 6):
            self.assertIn(f"액션-{index}", selected)
        for index in range(1, 7):
            self.assertIn(f"결정-{index}", selected)
        self.assertNotIn("잡담표식-ZZZ", selected)
        self.assertNotIn("[응답]", selected)

    def test_keeps_untagged_text(self) -> None:
        self.assertEqual(select_summary_sections("회의 전문"), "회의 전문")

    def test_selects_new_pipeline_tags_and_excludes_chitchat(self) -> None:
        text = "\n\n".join(
            [
                "CHITCHAT: 안녕하세요. 회의를 시작하겠습니다.",
                "INFO: 최근 중대재해 사례를 공유했습니다.",
                "DISCUSSION: 지게차 사고 예방 방안을 논의했습니다.",
                "QUESTION: 외국인 근로자 교육은 어떻게 진행합니까?",
                "ANSWER: 7개 국어 자료를 배포하겠습니다.",
                "DECISION: 안전 점검을 강화하기로 결정했습니다.",
                "ACTION: 각 청은 현장 점검 계획을 수립해 주세요.",
            ]
        )

        selected = select_summary_sections(text)

        self.assertNotIn("CHITCHAT:", selected)
        self.assertIn("INFO:", selected)
        self.assertIn("DISCUSSION:", selected)
        self.assertIn("QUESTION:", selected)
        self.assertIn("ANSWER:", selected)
        self.assertIn("DECISION:", selected)
        self.assertIn("ACTION:", selected)

    def test_selects_line_tagged_sections_and_excludes_chitchat(self) -> None:
        text = "\n".join(
            [
                "CHITCHAT: 안녕하세요. 회의를 시작하겠습니다.",
                "INFO: 최근 중대재해 사례를 공유했습니다.",
                "DISCUSSION: 지게차 사고 예방 방안을 논의했습니다.",
                "DECISION: 안전 점검을 강화하기로 결정했습니다.",
                "ACTION: 각 청은 현장 점검 계획을 수립해 주세요.",
                "CHITCHAT: 감사합니다. 회의를 마치겠습니다.",
            ]
        )

        selected = select_summary_sections(text)

        self.assertNotIn("CHITCHAT:", selected)
        self.assertNotIn("회의를 시작하겠습니다", selected)
        self.assertNotIn("회의를 마치겠습니다", selected)
        self.assertIn("INFO:", selected)
        self.assertIn("DISCUSSION:", selected)
        self.assertIn("DECISION:", selected)
        self.assertIn("ACTION:", selected)


class SummaryPromptPolicyTest(unittest.TestCase):
    def test_action_item_policy_preserves_unassigned_candidates(self) -> None:
        self.assertIn("담당자가 없어도", CHUNK_SYSTEM_PROMPT)
        self.assertIn("assignee_name을 null", CHUNK_SYSTEM_PROMPT)
        self.assertIn("임의 병합하거나 제거하지 마세요", CHUNK_SYSTEM_PROMPT)
        self.assertIn("제거하지 말고 duplicate_group_id로 묶으세요", MERGE_SYSTEM_PROMPT)
        self.assertIn("action_candidates", CHUNK_SYSTEM_PROMPT)
        self.assertIn("DB 저장용이 아닙니다", CHUNK_SYSTEM_PROMPT)
        self.assertIn("ACTION 태그라도 검토, 제안", CHUNK_SYSTEM_PROMPT)
        self.assertIn("QUESTION/ANSWER 태그 안에도", CHUNK_SYSTEM_PROMPT)
        self.assertIn("QUESTION/ANSWER 태그에서 나온 제안", MERGE_SYSTEM_PROMPT)

    def test_action_items_and_candidates_have_review_metadata(self) -> None:
        result = AnalyzeResponse.model_validate(VALID_RESULT)

        self.assertIs(result.action_items[0].source_type, ActionItemSourceType.DIRECTIVE)
        self.assertIs(result.action_items[0].confidence, ActionItemConfidence.HIGH)
        self.assertEqual(result.action_items[0].evidence, "김담당은 검색 API를 구현해 주세요.")
        self.assertIs(result.action_candidates[0].source_type, ActionItemSourceType.SUGGESTION)
        self.assertIs(result.action_candidates[0].confidence, ActionItemConfidence.MEDIUM)
        self.assertEqual(result.action_candidates[0].duplicate_group_id, "dup-1")

    def test_review_metadata_supports_all_frontend_filter_values(self) -> None:
        self.assertEqual(
            {item.value for item in ActionItemSourceType},
            {"DIRECTIVE", "PLAN", "SUGGESTION", "DISCUSSION"},
        )
        self.assertEqual(
            {item.value for item in ActionItemConfidence},
            {"HIGH", "MEDIUM", "LOW"},
        )

    def test_review_metadata_is_described_as_non_db_fields(self) -> None:
        fields = AiReviewActionItemData.model_fields

        for name in (
            "source_type",
            "confidence",
            "evidence",
            "duplicate_group_id",
        ):
            self.assertIn("DB 저장용 아님", fields[name].description)

    def test_date_policy_allows_only_explicit_or_current_date_relative_values(
        self,
    ) -> None:
        self.assertIn("현재 날짜 기준", CHUNK_SYSTEM_PROMPT)
        self.assertIn("오늘", CHUNK_SYSTEM_PROMPT)
        self.assertIn("내일", CHUNK_SYSTEM_PROMPT)
        self.assertIn("회의일, 월초, 월말, 분기말, 연말", CHUNK_SYSTEM_PROMPT)
        self.assertIn("새로 만들지 마세요", MERGE_SYSTEM_PROMPT)


class AnalyzeMeetingTest(unittest.IsolatedAsyncioTestCase):
    async def test_analyzes_chunks_then_merges_once(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
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
            patch(
                "app.services.summary_service._current_date_text",
                return_value="2026-06-26",
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
        self.assertIn("현재 날짜: 2026-06-26", first_chunk["messages"][1]["content"])
        self.assertIs(first_chunk["response_format"], ChunkAnalysis)
        self.assertIs(second_chunk["response_format"], ChunkAnalysis)
        self.assertIs(merge_request["response_format"], AnalyzeResponse)
        self.assertIn("검색 API를 구현한다.", merge_request["messages"][1]["content"])
        self.assertIn("관리자 권한 정책을 검토한다.", merge_request["messages"][1]["content"])
        self.assertIn('"current_date": "2026-06-26"', merge_request["messages"][1]["content"])

    async def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SummaryServiceError):
                await analyze_meeting("회의 전문")

    async def test_debug_response_includes_chunk_results(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
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
        parse = AsyncMock(side_effect=[chunk_response, final_response])
        client = MagicMock()
        client.chat.completions.parse = parse

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.summary_service._chunk_text",
                return_value=["하나의 구간"],
            ),
        ):
            result = await analyze_meeting_debug("회의 전문")

        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.selected_summary_text, "회의 전문")
        self.assertEqual(
            result.chunks[0].summary.objective,
            "신규 기능의 출시 범위를 정한다.",
        )
        self.assertEqual(result.result.meeting_summary, VALID_RESULT["meeting_summary"])

    async def test_debug_response_exposes_priority_tagged_summary_text(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
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
        parse = AsyncMock(side_effect=[chunk_response, final_response])
        client = MagicMock()
        client.chat.completions.parse = parse
        transcript = "\n\n".join(
            [
                "[의전/잡담] 잡담표식-ZZZ 인사말입니다.",
                "[핵심논의] 핵심논의-1 검색 기능 출시 범위를 논의했습니다.",
                "[결정사항] 결정-1 검색 기능은 MVP에 포함합니다.",
                "[액션후보] 액션-1 민수님은 검색 API QA 체크리스트를 작성해 주세요.",
            ]
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.summary_service._chunk_text",
                return_value=["선택된 하나의 구간"],
            ),
        ):
            result = await analyze_meeting_debug(transcript)

        self.assertIn("핵심논의-1", result.selected_summary_text)
        self.assertIn("결정-1", result.selected_summary_text)
        self.assertIn("액션-1", result.selected_summary_text)
        self.assertNotIn("잡담표식-ZZZ", result.selected_summary_text)

    async def test_strips_action_item_dates_without_source_evidence(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
            }
        )
        final_payload = {
            **VALID_RESULT,
            "action_items": [
                {
                    **VALID_RESULT["action_items"][0],
                    "due_date": "2026-06-30",
                    "evidence": "김담당은 검색 API를 구현해 주세요.",
                }
            ],
        }
        final_result = AnalyzeResponse.model_validate(final_payload)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch("app.services.summary_service._chunk_text", return_value=["ACTION: 김담당은 검색 API를 구현해 주세요."]),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting("ACTION: 김담당은 검색 API를 구현해 주세요.")

        self.assertIsNone(result.action_items[0].due_date)

    async def test_keeps_relative_today_tomorrow_dates_with_current_date_evidence(
        self,
    ) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
            }
        )
        final_payload = {
            **VALID_RESULT,
            "action_items": [
                {
                    **VALID_RESULT["action_items"][0],
                    "due_date": "2026-06-27",
                    "evidence": "김담당은 내일까지 검색 API를 구현해 주세요.",
                }
            ],
        }
        final_result = AnalyzeResponse.model_validate(final_payload)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch("app.services.summary_service._chunk_text", return_value=["ACTION: 김담당은 내일까지 검색 API를 구현해 주세요."]),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting("ACTION: 김담당은 내일까지 검색 API를 구현해 주세요.")

        self.assertEqual(result.action_items[0].due_date.isoformat(), "2026-06-27")

    async def test_tagged_text_without_decision_tag_does_not_invent_decision(
        self,
    ) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": VALID_RESULT["action_items"],
                "action_candidates": VALID_RESULT["action_candidates"],
            }
        )
        final_payload = {
            **VALID_RESULT,
            "summary": {
                **VALID_RESULT["summary"],
                "decision": "검색 기능을 출시하기로 결정했습니다.",
            },
        }
        final_result = AnalyzeResponse.model_validate(final_payload)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.summary_service._chunk_text",
                return_value=["ACTION: 김담당은 검색 API를 구현해 주세요."],
            ),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting("ACTION: 김담당은 검색 API를 구현해 주세요.")

        self.assertEqual(result.summary.decision, "명확한 결정사항 없음")

    async def test_operational_decision_tag_does_not_count_as_business_decision(
        self,
    ) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": [],
                "action_candidates": VALID_RESULT["action_candidates"],
            }
        )
        final_payload = {
            **VALID_RESULT,
            "summary": {
                **VALID_RESULT["summary"],
                "decision": "두 개 청만 발표를 진행하기로 결정했습니다.",
            },
            "action_items": [],
        }
        final_result = AnalyzeResponse.model_validate(final_payload)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])
        transcript = (
            "DECISION: 시간 관계상 모든 청에서 2분기 중점 추진 계획 발표는 어렵고 "
            "한 2개 청 정도에서만 발표를 진행하도록 하겠습니다."
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch("app.services.summary_service._chunk_text", return_value=[transcript]),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting(transcript)

        self.assertEqual(result.summary.decision, "명확한 결정사항 없음")

    async def test_business_decision_tag_keeps_decision_summary(self) -> None:
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": [],
                "action_candidates": [],
            }
        )
        final_result = AnalyzeResponse.model_validate(
            {
                **VALID_RESULT,
                "summary": {
                    **VALID_RESULT["summary"],
                    "decision": "검색 기능은 MVP 범위에 포함하기로 결정했습니다.",
                },
                "action_items": [],
                "action_candidates": [],
            }
        )
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])
        transcript = "DECISION: 검색 기능은 MVP 범위에 포함하기로 결정했습니다."

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch("app.services.summary_service._chunk_text", return_value=[transcript]),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting(transcript)

        self.assertEqual(
            result.summary.decision,
            "검색 기능은 MVP 범위에 포함하기로 결정했습니다.",
        )

    async def test_weak_action_items_are_reclassified_as_candidates(self) -> None:
        weak_item = {
            **VALID_RESULT["action_items"][0],
            "task": "현장 밀착 점검을 강화할 수 있도록 함께 노력할 것",
            "evidence": "현장 밀착 점검을 강화할 수 있도록 본부 지방관서와 함께 노력해야 될 때입니다.",
        }
        strong_item = {
            **VALID_RESULT["action_items"][0],
            "task": "신규 감독관 교육 자료를 제공하고 우수 자료를 공유할 것",
            "evidence": "본부에서 자료를 제공해 주시거나 우수한 자료들을 공유해 줄 수 있으시면 감사하겠습니다.",
        }
        chunk_result = ChunkAnalysis.model_validate(
            {
                "summary": VALID_RESULT["summary"],
                "action_items": [weak_item, strong_item],
                "action_candidates": [],
            }
        )
        final_payload = {
            **VALID_RESULT,
            "action_items": [weak_item, strong_item],
            "action_candidates": [],
        }
        final_result = AnalyzeResponse.model_validate(final_payload)
        chunk_response = MagicMock()
        chunk_response.choices = [
            MagicMock(message=MagicMock(parsed=chunk_result, content=None))
        ]
        final_response = MagicMock()
        final_response.choices = [
            MagicMock(message=MagicMock(parsed=final_result, content=None))
        ]
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(side_effect=[chunk_response, final_response])

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("app.services.summary_service.AsyncOpenAI", return_value=client),
            patch(
                "app.services.summary_service._chunk_text",
                return_value=[
                    "ACTION: 현장 밀착 점검을 강화할 수 있도록 함께 노력해야 합니다.\n"
                    "ACTION: 신규 감독관 교육 자료를 제공하고 우수 자료를 공유해 주세요."
                ],
            ),
            patch("app.services.summary_service._current_date_text", return_value="2026-06-26"),
        ):
            result = await analyze_meeting(
                "ACTION: 현장 밀착 점검을 강화할 수 있도록 함께 노력해야 합니다.\n"
                "ACTION: 신규 감독관 교육 자료를 제공하고 우수 자료를 공유해 주세요."
            )

        self.assertEqual(len(result.action_items), 1)
        self.assertEqual(
            result.action_items[0].task,
            "신규 감독관 교육 자료를 제공하고 우수 자료를 공유할 것",
        )
        self.assertEqual(len(result.action_candidates), 1)
        self.assertEqual(
            result.action_candidates[0].task,
            "현장 밀착 점검을 강화할 수 있도록 함께 노력할 것",
        )
        self.assertIs(result.action_candidates[0].source_type, ActionItemSourceType.SUGGESTION)
        self.assertIs(result.action_candidates[0].confidence, ActionItemConfidence.MEDIUM)


if __name__ == "__main__":
    unittest.main()
