import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile

from app.services.stt_service import (
    debug_transcribe_audio_bytes,
    transcribe_audio,
    transcribe_audio_bytes,
)
from app.services.summary_service import SummaryServiceError


class SttServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcribes_uploaded_audio_with_local_whisper_model(self) -> None:
        upload = UploadFile(filename="meeting.mp3", file=MagicMock())
        upload.file.read.return_value = b"audio-bytes"
        segment_one = SimpleNamespace(text=" 첫 번째 문장 ")
        segment_two = SimpleNamespace(text="두 번째 문장")

        with (
            patch("app.services.stt_service._get_whisper_model") as get_model,
            patch.dict("os.environ", {"STT_PROVIDER": "local", "STT_LANGUAGE": "ko"}),
        ):
            get_model.return_value.transcribe.return_value = (
                [segment_one, segment_two],
                object(),
            )
            result = await transcribe_audio(upload)

        self.assertEqual(result, "첫 번째 문장 두 번째 문장")
        get_model.return_value.transcribe.assert_called_once()
        transcribe_path = get_model.return_value.transcribe.call_args.args[0]
        self.assertTrue(transcribe_path.endswith(".mp3"))
        self.assertEqual(
            get_model.return_value.transcribe.call_args.kwargs,
            {"language": "ko", "vad_filter": True},
        )

    async def test_transcribes_audio_with_openai_provider(self) -> None:
        upload = UploadFile(filename="meeting.wav", file=MagicMock())
        upload.file.read.return_value = b"audio-bytes"
        client = MagicMock()
        client.audio.transcriptions.create = AsyncMock(
            return_value=SimpleNamespace(text=" 회의 음성 텍스트 ")
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "STT_PROVIDER": "openai",
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_STT_MODEL": "whisper-1",
                    "STT_LANGUAGE": "ko",
                },
                clear=True,
            ),
            patch("app.services.stt_service.AsyncOpenAI", return_value=client) as openai,
            patch("app.services.stt_service._get_whisper_model") as get_model,
        ):
            result = await transcribe_audio(upload)

        self.assertEqual(result, "회의 음성 텍스트")
        openai.assert_called_once_with(api_key="test-key")
        get_model.assert_not_called()
        create = client.audio.transcriptions.create
        create.assert_awaited_once()
        self.assertEqual(create.await_args.kwargs["model"], "whisper-1")
        self.assertEqual(create.await_args.kwargs["language"], "ko")
        self.assertEqual(create.await_args.kwargs["file"].name, "audio.wav")

    async def test_openai_provider_requires_api_key(self) -> None:
        with patch.dict("os.environ", {"STT_PROVIDER": "openai"}, clear=True):
            with self.assertRaises(SummaryServiceError):
                await transcribe_audio_bytes(b"audio-bytes", suffix=".mp3")

    async def test_rejects_unknown_stt_provider(self) -> None:
        with patch.dict("os.environ", {"STT_PROVIDER": "invalid"}, clear=True):
            with self.assertRaises(SummaryServiceError):
                await transcribe_audio_bytes(b"audio-bytes", suffix=".mp3")

    async def test_debug_transcribes_same_audio_with_both_providers(self) -> None:
        with (
            patch(
                "app.services.stt_service._transcribe_local_bytes",
                new=AsyncMock(return_value="로컬 결과"),
            ) as local,
            patch(
                "app.services.stt_service._transcribe_openai",
                new=AsyncMock(return_value="오픈AI 결과"),
            ) as openai,
        ):
            result = await debug_transcribe_audio_bytes(b"audio-bytes", suffix=".wav")

        self.assertTrue(result.local.available)
        self.assertTrue(result.openai.available)
        self.assertEqual(result.local.text, "로컬 결과")
        self.assertEqual(result.openai.text, "오픈AI 결과")
        self.assertIsInstance(result.local.elapsed_ms, int)
        self.assertIsInstance(result.openai.elapsed_ms, int)
        local.assert_awaited_once_with(b"audio-bytes", ".wav")
        openai.assert_awaited_once_with(b"audio-bytes", ".wav")

    async def test_debug_reports_provider_error_without_failing_entire_comparison(
        self,
    ) -> None:
        with (
            patch(
                "app.services.stt_service._transcribe_local_bytes",
                new=AsyncMock(return_value="로컬 결과"),
            ),
            patch(
                "app.services.stt_service._transcribe_openai",
                new=AsyncMock(side_effect=SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")),
            ),
        ):
            result = await debug_transcribe_audio_bytes(b"audio-bytes", suffix=".wav")

        self.assertTrue(result.local.available)
        self.assertFalse(result.openai.available)
        self.assertEqual(result.openai.error, "OPENAI_API_KEY가 설정되지 않았습니다.")
        self.assertIsNone(result.openai.text)

    async def test_rejects_empty_audio_file(self) -> None:
        upload = UploadFile(filename="empty.mp3", file=MagicMock())
        upload.file.read.return_value = b""

        with self.assertRaises(SummaryServiceError):
            await transcribe_audio(upload)


if __name__ == "__main__":
    unittest.main()
