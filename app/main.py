from json import JSONDecodeError

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import ValidationError

from app.schemas import (
    AnalyzeDebugResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    PreprocessDebugResponse,
    PreprocessResponse,
    ShortSummaryResponse,
    SttDebugResponse,
    SttJobCreateResponse,
    SttJobRequest,
    SttJobStatusResponse,
    TaggingDebugResponse,
    TaggingResponse,
    TextResponse,
)
from app.services.preprocess_service import (
    debug_preprocess_meeting_transcript,
    debug_tag_meeting_transcript,
    preprocess_meeting_transcripts,
    preprocess_transcript,
    preprocess_meeting_transcript,
    tag_meeting_transcript,
)
from app.services.storage_service import default_audio_bucket
from app.services.stt_job_service import create_stt_job, get_stt_job, process_stt_job
from app.services.stt_service import debug_transcribe_audio, transcribe_audio
from app.services.summary_service import (
    SummaryServiceError,
    analyze_meeting,
    analyze_meeting_debug,
)

load_dotenv("/app/.env")

app = FastAPI(title="Damlok AI API")

TEXT_REQUEST_BODY_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/AnalyzeRequest"},
                "example": {"text": "회의 전문을 여기에 입력합니다."},
            },
            "text/plain": {
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "description": "줄바꿈을 포함한 회의 전문 텍스트",
                },
                "example": "회의 전문을 여기에 입력합니다.",
            },
        },
    }
}

STT_FILE_REQUEST_BODY_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "업로드할 음성 파일",
                        }
                    },
                }
            }
        },
    }
}


async def _analyze_text(text: str) -> AnalyzeResponse:
    try:
        return await analyze_meeting(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="회의 분석을 생성하지 못했습니다.",
        ) from exc


async def _analyze_text_debug(text: str) -> AnalyzeDebugResponse:
    try:
        return await analyze_meeting_debug(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="회의 분석을 생성하지 못했습니다.",
        ) from exc


async def _preprocess_text(text: str) -> PreprocessResponse:
    try:
        return await preprocess_meeting_transcripts(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"회의 전문을 전처리하지 못했습니다: {exc}",
        ) from exc


async def _debug_preprocess_text(text: str) -> PreprocessDebugResponse:
    try:
        return await debug_preprocess_meeting_transcript(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"회의 전문을 전처리하지 못했습니다: {exc}",
        ) from exc


async def _tagging_text(text: str) -> TaggingResponse:
    try:
        return await tag_meeting_transcript(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"회의 전문을 태깅하지 못했습니다: {exc}",
        ) from exc


async def _debug_tagging_text(text: str) -> TaggingDebugResponse:
    try:
        return await debug_tag_meeting_transcript(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"회의 전문을 태깅하지 못했습니다: {exc}",
        ) from exc


async def _extract_text(request: Request) -> str:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("text/plain"):
        text = (await request.body()).decode("utf-8")
    else:
        try:
            payload = await request.json()
            text = AnalyzeRequest.model_validate(payload).text
        except (JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status_code=422,
                detail='JSON 본문은 {"text": "..."} 형식이어야 합니다.',
            ) from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="분석할 텍스트가 비어 있습니다.")
    return text


def _clean_for_summary(text: str) -> str:
    return preprocess_transcript(text)


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    summary="회의 요약 및 액션아이템 분석",
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return await _analyze_text(request.text)


@app.post(
    "/api/analyze/text",
    response_model=AnalyzeResponse,
    summary="긴 회의 전문 분석 테스트",
)
async def analyze_plain_text(
    text: str = Body(
        min_length=1,
        media_type="text/plain",
        description="줄바꿈을 포함한 회의 전문 원문",
    ),
) -> AnalyzeResponse:
    return await _analyze_text(text)


@app.post(
    "/aiactions/stt",
    response_model=TextResponse,
    summary="프로젝트 회의록 음성 텍스트화",
    openapi_extra=STT_FILE_REQUEST_BODY_DOC,
)
async def aiactions_stt(file: UploadFile = File(...)) -> TextResponse:
    try:
        return TextResponse(text=await transcribe_audio(file))
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="회의 음성을 텍스트로 변환하지 못했습니다.",
        ) from exc


@app.post(
    "/aiactions/stt/debug",
    response_model=SttDebugResponse,
    summary="프로젝트 회의록 STT provider 속도 비교",
    openapi_extra=STT_FILE_REQUEST_BODY_DOC,
)
async def aiactions_stt_debug(file: UploadFile = File(...)) -> SttDebugResponse:
    try:
        return await debug_transcribe_audio(file)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="회의 음성 STT provider 비교를 수행하지 못했습니다.",
        ) from exc


@app.post(
    "/aiactions/stt/jobs",
    response_model=SttJobCreateResponse,
    summary="Supabase Storage 음성 파일 STT job 생성",
)
async def create_aiactions_stt_job(
    request: SttJobRequest,
    background_tasks: BackgroundTasks,
) -> SttJobCreateResponse:
    bucket = request.bucket or default_audio_bucket()
    job = create_stt_job(bucket=bucket, path=request.path)
    background_tasks.add_task(process_stt_job, job.job_id)
    return SttJobCreateResponse(job_id=job.job_id, status=job.status)


@app.get(
    "/aiactions/stt/jobs/{job_id}",
    response_model=SttJobStatusResponse,
    summary="Supabase Storage 음성 파일 STT job 상태 조회",
)
async def get_aiactions_stt_job(job_id: str) -> SttJobStatusResponse:
    job = get_stt_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="STT job을 찾을 수 없습니다.")

    return SttJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        text=job.text,
        error=job.error,
    )


@app.post(
    "/aiactions/preprocess",
    response_model=PreprocessResponse,
    summary="프로젝트 회의록 텍스트 전처리",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_preprocess(request: Request) -> PreprocessResponse:
    return await _preprocess_text(await _extract_text(request))


@app.post(
    "/aiactions/preprocess/debug",
    response_model=PreprocessDebugResponse,
    summary="프로젝트 회의록 텍스트 전처리 디버그",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_preprocess_debug(request: Request) -> PreprocessDebugResponse:
    return await _debug_preprocess_text(await _extract_text(request))


@app.post(
    "/aiactions/tagging",
    response_model=TaggingResponse,
    summary="프로젝트 회의록 텍스트 섹션 태깅",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_tagging(request: Request) -> TaggingResponse:
    return await _tagging_text(await _extract_text(request))


@app.post(
    "/aiactions/tagging/debug",
    response_model=TaggingDebugResponse,
    summary="프로젝트 회의록 텍스트 섹션 태깅 디버그",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_tagging_debug(request: Request) -> TaggingDebugResponse:
    return await _debug_tagging_text(await _extract_text(request))


@app.post(
    "/aiactions/summary/full",
    response_model=AnalyzeResponse,
    summary="프로젝트 회의록 텍스트 AI 전체 요약",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_summary_full(request: Request) -> AnalyzeResponse:
    return await _analyze_text(_clean_for_summary(await _extract_text(request)))


@app.post(
    "/aiactions/summary/short",
    response_model=ShortSummaryResponse,
    summary="프로젝트 회의록 텍스트 AI 한 줄 요약",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_summary_short(request: Request) -> ShortSummaryResponse:
    result = await _analyze_text(_clean_for_summary(await _extract_text(request)))
    return ShortSummaryResponse(meeting_summary=result.meeting_summary)


@app.post(
    "/aiactions/summary/debug",
    response_model=AnalyzeDebugResponse,
    summary="프로젝트 회의록 텍스트 AI 청크 분석 디버그",
    openapi_extra=TEXT_REQUEST_BODY_DOC,
)
async def aiactions_summary_debug(request: Request) -> AnalyzeDebugResponse:
    return await _analyze_text_debug(_clean_for_summary(await _extract_text(request)))


@app.patch("/aiactions/upadate/full", summary="프로젝트 회의록 전체 요약 수정")
async def aiactions_update_full(mid: int) -> None:
    raise HTTPException(status_code=501, detail="Java 서버에서 처리하는 기능입니다.")


@app.patch("/aiactions/upadate/short", summary="프로젝트 회의록 한 줄 요약 수정")
async def aiactions_update_short(mid: int) -> None:
    raise HTTPException(status_code=501, detail="Java 서버에서 처리하는 기능입니다.")


@app.patch("/aiactions/upadate/action", summary="프로젝트 회의록 action item 수정")
async def aiactions_update_action(mid: int) -> None:
    raise HTTPException(status_code=501, detail="Java 서버에서 처리하는 기능입니다.")
