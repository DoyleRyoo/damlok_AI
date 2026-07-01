import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    AnalyzeDebugResponse,
    AnalyzeResponse,
    PreprocessDebugResponse,
    PreprocessResponse,
    SttDebugResponse,
    TaggingDebugResponse,
    TaggingResponse,
)
from app.services.stt_job_service import _JOBS
from app.services.summary_service import SummaryServiceError


RESULT = AnalyzeResponse.model_validate(
    {
        "summary": {
            "objective": "테스트 목적",
            "discussion": "여러 줄의 논의",
            "decision": "테스트 결정",
        },
        "action_items": [],
        "action_candidates": [],
        "meeting_summary": "긴 회의 원문 분석 테스트",
    }
)

STT_DEBUG_RESULT = SttDebugResponse.model_validate(
    {
        "local": {
            "available": True,
            "elapsed_ms": 1200,
            "text": "로컬 STT 결과",
            "error": None,
        },
        "openai": {
            "available": True,
            "elapsed_ms": 800,
            "text": "OpenAI STT 결과",
            "error": None,
        },
    }
)

DEBUG_RESULT = AnalyzeDebugResponse.model_validate(
    {
        "chunk_count": 1,
        "selected_summary_text": "[핵심논의] 회의 전문",
        "chunks": [
            {
                "summary": {
                    "objective": "테스트 목적",
                    "discussion": "청크 논의",
                    "decision": "청크 결정",
                },
                "action_items": [],
                "action_candidates": [],
            }
        ],
        "result": RESULT.model_dump(mode="json"),
    }
)

PREPROCESS_DEBUG_RESULT = PreprocessDebugResponse(
    mode="basic",
    raw_transcript="  어 원문  ",
    cleaned_transcript="어 원문",
    input_chars=8,
    cleaned_chars=4,
    removed_chars=4,
    basic_chars=4,
    ai_chars=None,
    raw_ai_chars=None,
    final_processed_chars=4,
    total_chunks=0,
    ai_used_chunks=0,
    fallback_chunks=0,
    too_short_count=0,
    too_long_count=0,
    preserved_dates=True,
    preserved_numbers=True,
    preserved_entities=True,
    preserved_directives=True,
    date_preserve_ratio=1.0,
    number_preserve_ratio=1.0,
    entity_preserve_ratio=1.0,
    directive_preserve_ratio=1.0,
    section_counts={},
    action_candidate_count=0,
    decision_candidate_count=0,
    ignored_section_count=0,
    fallback_reason_by_chunk=[],
    fallback_used=False,
    fallback_reason=None,
    basic_preview="어 원문",
    ai_preview=None,
)

PREPROCESS_RESULT = PreprocessResponse(
    text="첫 줄 입니다.\n\n둘째 줄",
    raw_transcript="  첫 줄\t입니다.\r\n\r\n\r\n  둘째 줄  ",
    cleaned_transcript="첫 줄 입니다.\n\n둘째 줄",
    readable_transcript="첫 줄입니다. 둘째 줄입니다.",
    readable_available=True,
    readable_error=None,
)

TAGGING_RESULT = TaggingResponse(
    text="ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.",
    raw_transcript="  서울청장은 현장 점검 계획을 수립해 주세요.  ",
    cleaned_transcript="서울청장은 현장 점검 계획을 수립해 주세요.",
    tagged_transcript="ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.",
)

TAGGING_DEBUG_RESULT = TaggingDebugResponse(
    raw_transcript="회의 원문",
    cleaned_transcript="회의 원문",
    tagged_transcript="\n\n".join(
        [
            "ACTION: 현장 점검 계획을 수립해 주세요.",
            "DECISION: 안전 점검을 강화하기로 했습니다.",
            "CHITCHAT: 회의를 시작하겠습니다.",
        ]
    ),
    tag_counts={"ACTION": 1, "DECISION": 1, "CHITCHAT": 1},
    summary_used_tag_counts={"ACTION": 1, "DECISION": 1},
    ignored_tag_counts={"CHITCHAT": 1},
    action_count=1,
    decision_count=1,
    discussion_count=0,
    chitchat_count=1,
)


class AnalyzePlainTextApiTest(unittest.TestCase):
    def test_accepts_multiline_plain_text_without_json_escaping(self) -> None:
        transcript = '첫 번째 발언입니다.\n"인용문"이 포함된 두 번째 발언입니다.'

        with patch(
            "app.main.analyze_meeting",
            new=AsyncMock(return_value=RESULT),
        ) as analyze:
            response = TestClient(app).post(
                "/api/analyze/text",
                content=transcript.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meeting_summary"], RESULT.meeting_summary)
        analyze.assert_awaited_once_with(transcript)


class AiActionsApiTest(unittest.TestCase):
    def setUp(self) -> None:
        _JOBS.clear()

    def test_stt_accepts_multipart_file_and_returns_transcribed_text(self) -> None:
        contents = b"audio-bytes"

        with patch(
            "app.main.transcribe_audio",
            new=AsyncMock(return_value="회의 음성 텍스트입니다."),
        ) as transcribe:
            response = TestClient(app).post(
                "/aiactions/stt",
                files={"file": ("meeting.wav", contents, "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "회의 음성 텍스트입니다."})
        uploaded_file = transcribe.await_args.args[0]
        self.assertEqual(uploaded_file.filename, "meeting.wav")
        self.assertEqual(uploaded_file.content_type, "audio/wav")

    def test_stt_debug_compares_local_and_openai_providers(self) -> None:
        contents = b"audio-bytes"

        with patch(
            "app.main.debug_transcribe_audio",
            new=AsyncMock(return_value=STT_DEBUG_RESULT),
        ) as debug_transcribe:
            response = TestClient(app).post(
                "/aiactions/stt/debug",
                files={"file": ("meeting.wav", contents, "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["local"]["elapsed_ms"], 1200)
        self.assertEqual(response.json()["openai"]["elapsed_ms"], 800)
        self.assertEqual(response.json()["local"]["text"], "로컬 STT 결과")
        self.assertEqual(response.json()["openai"]["text"], "OpenAI STT 결과")
        uploaded_file = debug_transcribe.await_args.args[0]
        self.assertEqual(uploaded_file.filename, "meeting.wav")
        self.assertEqual(uploaded_file.content_type, "audio/wav")

    def test_stt_openapi_uses_multipart_form_data(self) -> None:
        operation = app.openapi()["paths"]["/aiactions/stt"]["post"]
        content = operation["requestBody"]["content"]

        self.assertIn("multipart/form-data", content)
        schema = content["multipart/form-data"]["schema"]
        if "$ref" in schema and "properties" not in schema:
            schema_name = schema["$ref"].rsplit("/", 1)[-1]
            schema = app.openapi()["components"]["schemas"][schema_name]
        self.assertEqual(schema["properties"]["file"]["format"], "binary")

    def test_creates_storage_stt_job(self) -> None:
        with patch(
            "app.main.process_stt_job",
            new=AsyncMock(),
        ) as process:
            response = TestClient(app).post(
                "/aiactions/stt/jobs",
                json={"path": "meetings/1/audio.mp3"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "PENDING")
        self.assertIn("job_id", response.json())
        process.assert_awaited_once_with(response.json()["job_id"])

    def test_gets_storage_stt_job_status(self) -> None:
        with patch("app.main.process_stt_job", new=AsyncMock()):
            create_response = TestClient(app).post(
                "/aiactions/stt/jobs",
                json={"bucket": "meeting-audio", "path": "meetings/1/audio.mp3"},
            )

        job_id = create_response.json()["job_id"]
        response = TestClient(app).get(f"/aiactions/stt/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], job_id)
        self.assertEqual(response.json()["status"], "PENDING")
        self.assertIsNone(response.json()["text"])
        self.assertIsNone(response.json()["error"])

    def test_missing_storage_stt_job_returns_404(self) -> None:
        response = TestClient(app).get("/aiactions/stt/jobs/missing")

        self.assertEqual(response.status_code, 404)

    def test_full_summary_accepts_plain_text(self) -> None:
        transcript = "긴 회의 전문입니다.\n결정사항이 포함됩니다."

        with patch(
            "app.main.analyze_meeting",
            new=AsyncMock(return_value=RESULT),
        ) as analyze:
            response = TestClient(app).post(
                "/aiactions/summary/full",
                content=transcript.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["objective"], "테스트 목적")
        analyze.assert_awaited_once_with(transcript)

    def test_full_summary_uses_basic_preprocessed_text(self) -> None:
        transcript = "  긴 회의 전문입니다.\r\n\r\n\r\n결정사항이 포함됩니다.  "

        with patch(
            "app.main.analyze_meeting",
            new=AsyncMock(return_value=RESULT),
        ) as analyze:
            response = TestClient(app).post(
                "/aiactions/summary/full",
                content=transcript.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        self.assertEqual(response.status_code, 200)
        analyze.assert_awaited_once_with("긴 회의 전문입니다.\n\n결정사항이 포함됩니다.")

    def test_short_summary_returns_only_meeting_summary(self) -> None:
        with patch("app.main.analyze_meeting", new=AsyncMock(return_value=RESULT)):
            response = TestClient(app).post(
                "/aiactions/summary/short",
                json={"text": "회의 전문"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"meeting_summary": RESULT.meeting_summary})

    def test_preprocess_normalizes_transcript_text(self) -> None:
        with patch(
            "app.main.preprocess_meeting_transcripts",
            new=AsyncMock(return_value=PREPROCESS_RESULT),
        ):
            response = TestClient(app).post(
                "/aiactions/preprocess",
                json={"text": PREPROCESS_RESULT.raw_transcript},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": "첫 줄 입니다.\n\n둘째 줄",
                "raw_transcript": "  첫 줄\t입니다.\r\n\r\n\r\n  둘째 줄  ",
                "cleaned_transcript": "첫 줄 입니다.\n\n둘째 줄",
                "readable_transcript": "첫 줄입니다. 둘째 줄입니다.",
                "readable_available": True,
                "readable_error": None,
            },
        )

    def test_preprocess_returns_502_when_preprocess_fails(self) -> None:
        with patch(
            "app.main.preprocess_meeting_transcripts",
            new=AsyncMock(side_effect=SummaryServiceError("boom")),
        ):
            response = TestClient(app).post(
                "/aiactions/preprocess",
                json={"text": "회의 전문"},
            )

        self.assertEqual(response.status_code, 502)

    def test_tagging_returns_tagged_transcript(self) -> None:
        with patch(
            "app.main.tag_meeting_transcript",
            new=AsyncMock(return_value=TAGGING_RESULT),
        ) as tagging:
            response = TestClient(app).post(
                "/aiactions/tagging",
                json={"text": TAGGING_RESULT.raw_transcript},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": "ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.",
                "raw_transcript": "  서울청장은 현장 점검 계획을 수립해 주세요.  ",
                "cleaned_transcript": "서울청장은 현장 점검 계획을 수립해 주세요.",
                "tagged_transcript": "ACTION: 서울청장은 현장 점검 계획을 수립해 주세요.",
            },
        )
        tagging.assert_awaited_once_with(TAGGING_RESULT.raw_transcript)

    def test_tagging_debug_returns_pipeline_tag_counts(self) -> None:
        with patch(
            "app.main.debug_tag_meeting_transcript",
            new=AsyncMock(return_value=TAGGING_DEBUG_RESULT),
        ) as tagging:
            response = TestClient(app).post(
                "/aiactions/tagging/debug",
                json={"text": TAGGING_DEBUG_RESULT.raw_transcript},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tag_counts"], {"ACTION": 1, "DECISION": 1, "CHITCHAT": 1})
        self.assertEqual(response.json()["summary_used_tag_counts"], {"ACTION": 1, "DECISION": 1})
        self.assertEqual(response.json()["ignored_tag_counts"], {"CHITCHAT": 1})
        self.assertEqual(response.json()["action_count"], 1)
        self.assertEqual(response.json()["chitchat_count"], 1)
        tagging.assert_awaited_once_with(TAGGING_DEBUG_RESULT.raw_transcript)

    def test_preprocess_debug_returns_cleanup_only_result(self) -> None:
        with patch(
            "app.main.debug_preprocess_meeting_transcript",
            new=AsyncMock(return_value=PREPROCESS_DEBUG_RESULT),
        ):
            response = TestClient(app).post(
                "/aiactions/preprocess/debug",
                content="  어 원문  ".encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "mode": "basic",
                "raw_transcript": "  어 원문  ",
                "cleaned_transcript": "어 원문",
                "input_chars": 8,
                "cleaned_chars": 4,
                "removed_chars": 4,
                "basic_chars": 4,
                "ai_chars": None,
                "raw_ai_chars": None,
                "final_processed_chars": 4,
                "total_chunks": 0,
                "ai_used_chunks": 0,
                "fallback_chunks": 0,
                "too_short_count": 0,
                "too_long_count": 0,
                "preserved_dates": True,
                "preserved_numbers": True,
                "preserved_entities": True,
                "preserved_directives": True,
                "date_preserve_ratio": 1.0,
                "number_preserve_ratio": 1.0,
                "entity_preserve_ratio": 1.0,
                "directive_preserve_ratio": 1.0,
                "section_counts": {},
                "action_candidate_count": 0,
                "decision_candidate_count": 0,
                "ignored_section_count": 0,
                "fallback_reason_by_chunk": [],
                "fallback_used": False,
                "fallback_reason": None,
                "basic_preview": "어 원문",
                "ai_preview": None,
            },
        )

    def test_debug_summary_returns_chunk_results_and_final_result(self) -> None:
        with patch(
            "app.main.analyze_meeting_debug",
            new=AsyncMock(return_value=DEBUG_RESULT),
        ) as analyze:
            response = TestClient(app).post(
                "/aiactions/summary/debug",
                json={"text": "회의 전문"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunk_count"], 1)
        self.assertEqual(
            response.json()["selected_summary_text"],
            "[핵심논의] 회의 전문",
        )
        self.assertEqual(
            response.json()["chunks"][0]["summary"]["decision"],
            "청크 결정",
        )
        self.assertEqual(
            response.json()["result"]["meeting_summary"],
            RESULT.meeting_summary,
        )
        analyze.assert_awaited_once_with("회의 전문")

    def test_java_update_routes_are_not_implemented_in_ai_server(self) -> None:
        response = TestClient(app).patch("/aiactions/upadate/full?mid=3")

        self.assertEqual(response.status_code, 501)


if __name__ == "__main__":
    unittest.main()
