from fastapi import Body, FastAPI, HTTPException

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.summary_service import SummaryServiceError, analyze_meeting

app = FastAPI(title="Damlok AI API")


async def _analyze_text(text: str) -> AnalyzeResponse:
    try:
        return await analyze_meeting(text)
    except SummaryServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="회의 분석을 생성하지 못했습니다.",
        ) from exc


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
