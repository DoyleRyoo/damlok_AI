import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AnalyzeResponse


RESULT = AnalyzeResponse.model_validate(
    {
        "summary": {
            "objective": "테스트 목적",
            "discussion": "여러 줄의 논의",
            "decision": "테스트 결정",
        },
        "action_items": [],
        "meeting_summary": "긴 회의 원문 분석 테스트",
    }
)


class AnalyzePlainTextApiTest(unittest.TestCase):
    def test_accepts_multiline_plain_text_without_json_escaping(self) -> None:
        transcript = '첫 번째 발언입니다.\n"인용문"이 포함된 두 번째 발언입니다.'

        with patch("app.main.analyze_meeting", new=AsyncMock(return_value=RESULT)) as analyze:
            response = TestClient(app).post(
                "/api/analyze/text",
                content=transcript.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meeting_summary"], RESULT.meeting_summary)
        analyze.assert_awaited_once_with(transcript)


if __name__ == "__main__":
    unittest.main()
