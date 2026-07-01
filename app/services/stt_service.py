import asyncio
import io
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import UploadFile
from faster_whisper import WhisperModel
from openai import AsyncOpenAI, OpenAIError

from app.schemas import SttDebugResponse, SttProviderDebugResult
from app.services.summary_service import SummaryServiceError


DEFAULT_STT_MODEL = "small"
DEFAULT_STT_DEVICE = "cpu"
DEFAULT_STT_COMPUTE_TYPE = "int8"
DEFAULT_STT_PROVIDER = "local"
DEFAULT_OPENAI_STT_MODEL = "whisper-1"


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    model_name = os.getenv("STT_MODEL", DEFAULT_STT_MODEL)
    device = os.getenv("STT_DEVICE", DEFAULT_STT_DEVICE)
    compute_type = os.getenv("STT_COMPUTE_TYPE", DEFAULT_STT_COMPUTE_TYPE)

    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise SummaryServiceError("로컬 STT 모델을 초기화하지 못했습니다.") from exc


def _transcribe_path(path: str) -> str:
    model = _get_whisper_model()
    language = os.getenv("STT_LANGUAGE", "ko")

    try:
        segments, _ = model.transcribe(path, language=language, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments if segment.text)
    except Exception as exc:
        raise SummaryServiceError("로컬 STT 변환에 실패했습니다.") from exc

    if not text.strip():
        raise SummaryServiceError("STT 결과가 비어 있습니다.")
    return text.strip()


def _stt_provider() -> str:
    provider = os.getenv("STT_PROVIDER", DEFAULT_STT_PROVIDER).strip().lower()
    if provider not in {"local", "openai"}:
        raise SummaryServiceError("STT_PROVIDER는 local 또는 openai 중 하나여야 합니다.")
    return provider


async def _transcribe_openai(audio: bytes, suffix: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")

    language = os.getenv("STT_LANGUAGE", "ko")
    model_name = os.getenv("OPENAI_STT_MODEL", DEFAULT_OPENAI_STT_MODEL)
    filename = f"audio{suffix}"
    audio_file = io.BytesIO(audio)
    audio_file.name = filename

    try:
        client = AsyncOpenAI(api_key=api_key)
        result = await client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            language=language,
        )
    except OpenAIError as exc:
        raise SummaryServiceError("OpenAI STT 요청에 실패했습니다.") from exc

    text = getattr(result, "text", None)
    if not text or not text.strip():
        raise SummaryServiceError("STT 결과가 비어 있습니다.")
    return text.strip()


async def _transcribe_local_bytes(audio: bytes, suffix: str) -> str:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio)
            temp_path = temp_file.name

        return await asyncio.to_thread(_transcribe_path, temp_path)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


async def _measure_provider(provider: str, audio: bytes, suffix: str) -> SttProviderDebugResult:
    started = time.perf_counter()
    try:
        if provider == "local":
            text = await _transcribe_local_bytes(audio, suffix)
        elif provider == "openai":
            text = await _transcribe_openai(audio, suffix)
        else:
            raise SummaryServiceError("STT_PROVIDER는 local 또는 openai 중 하나여야 합니다.")
    except SummaryServiceError as exc:
        return SttProviderDebugResult(
            available=False,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            text=None,
            error=str(exc),
        )

    return SttProviderDebugResult(
        available=True,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        text=text,
        error=None,
    )


async def transcribe_audio(file: UploadFile) -> str:
    audio = await file.read()
    suffix = Path(file.filename or "").suffix or ".audio"
    return await transcribe_audio_bytes(audio, suffix=suffix)


async def transcribe_audio_bytes(audio: bytes, suffix: str = ".audio") -> str:
    if not audio:
        raise SummaryServiceError("음성 파일이 비어 있습니다.")

    if not suffix.startswith("."):
        suffix = f".{suffix}"

    provider = _stt_provider()
    if provider == "openai":
        return await _transcribe_openai(audio, suffix)

    return await _transcribe_local_bytes(audio, suffix)


async def debug_transcribe_audio(file: UploadFile) -> SttDebugResponse:
    audio = await file.read()
    suffix = Path(file.filename or "").suffix or ".audio"
    return await debug_transcribe_audio_bytes(audio, suffix=suffix)


async def debug_transcribe_audio_bytes(audio: bytes, suffix: str = ".audio") -> SttDebugResponse:
    if not audio:
        raise SummaryServiceError("음성 파일이 비어 있습니다.")

    if not suffix.startswith("."):
        suffix = f".{suffix}"

    local, openai = await asyncio.gather(
        _measure_provider("local", audio, suffix),
        _measure_provider("openai", audio, suffix),
    )
    return SttDebugResponse(local=local, openai=openai)
