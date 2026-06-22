import asyncio
import json
import os
from collections.abc import Sequence
from typing import TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError
import tiktoken

from app.schemas import AnalyzeResponse, ChunkAnalysis


CHUNK_SYSTEM_PROMPT = """
당신은 회의록 분석 전문가입니다.

입력은 긴 회의록을 나눈 일부 구간입니다.
제공된 구간만 근거로 상세 요약과 액션 아이템 후보를 추출하세요.
추측하지 말고, 텍스트에서 확인할 수 없는 내용은 null 또는 "확인되지 않음"으로 작성하세요.
다른 구간에서 최종 통합하므로 내용을 과도하게 축약하지 마세요.

반드시 다음 JSON 구조로만 응답하세요.

{
  "summary": {
    "objective": "회의 목적",
    "discussion": "핵심 논의 내용",
    "decision": "주요 결정 사항"
  },
  "action_items": [
    {
      "assignee_name": "담당자 이름 또는 null",
      "assignee_email": "담당자 이메일 또는 null",
      "task": "해야 할 일",
      "start_date": "YYYY-MM-DD 또는 null",
      "due_date": "YYYY-MM-DD 또는 null",
      "priority": "HIGH 또는 MEDIUM 또는 LOW",
      "status": "미착수"
    }
  ]
}

규칙:
- 이 구간에서 확인되는 내용만 구체적으로 작성하세요.
- objective에는 회의의 배경과 목적을, discussion에는 주요 쟁점과 의견을,
  decision에는 확정된 결정과 미결 사항을 구분해 작성하세요.
- action_items가 없으면 빈 배열 [] 로 작성하세요.
- 발언에서 실행 주체와 할 일이 드러난 항목만 action_items로 추출하세요.
- 담당자가 명확하지 않으면 assignee_name과 assignee_email을 null로 작성하세요.
- 날짜가 명확하지 않으면 null로 작성하세요.
- priority는 HIGH, MEDIUM, LOW 중 하나만 사용하세요.
- 완료 또는 진행 중이라고 명시되지 않은 액션 아이템의 status는 "미착수"로 작성하세요.
"""

MERGE_SYSTEM_PROMPT = """
당신은 여러 회의록 구간의 분석 결과를 하나로 통합하는 회의록 분석 전문가입니다.

입력으로 제공된 구간별 분석만 근거로 최종 결과를 작성하세요.
중복되는 내용과 액션 아이템은 하나로 합치고, 서로 다른 내용은 누락하지 마세요.
추측하거나 새로운 담당자, 날짜, 결정 사항을 추가하지 마세요.

규칙:
- summary는 회의 내용을 다시 확인하지 않아도 될 만큼 구체적으로 작성하세요.
- objective에는 전체 회의의 배경과 목적을 작성하세요.
- discussion에는 주요 쟁점, 의견, 근거와 미결 사항을 종합하세요.
- decision에는 실제로 확정된 결정 사항만 작성하세요.
- 동일한 task, 담당자, 날짜를 가진 액션 아이템은 하나로 합치세요.
- 담당자나 날짜가 확인되지 않은 값은 null을 유지하세요.
- meeting_summary는 상세 요약을 대신하지 않으며, 벡터 검색용 1문장으로 짧게 작성하세요.
"""

ModelT = TypeVar("ModelT", bound=BaseModel)
DEFAULT_CHUNK_SIZE = 6_000
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_MAX_CONCURRENCY = 3


class SummaryServiceError(Exception):
    pass


def _parse_result(content: str | None, response_model: type[ModelT]) -> ModelT:
    if not content:
        raise SummaryServiceError("모델 응답이 비어 있습니다.")

    try:
        return response_model.model_validate(json.loads(content))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise SummaryServiceError("모델 응답 형식이 올바르지 않습니다.") from exc


def _parse_analyze_result(content: str | None) -> AnalyzeResponse:
    return _parse_result(content, AnalyzeResponse)


def _positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise SummaryServiceError(f"{name} 설정이 올바르지 않습니다.") from exc

    if value <= 0:
        raise SummaryServiceError(f"{name} 설정은 0보다 커야 합니다.")
    return value


def _chunk_text(text: str, model_name: str) -> list[str]:
    chunk_size = _positive_int_setting("SUMMARY_CHUNK_TOKENS", DEFAULT_CHUNK_SIZE)
    overlap = _positive_int_setting("SUMMARY_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    if overlap >= chunk_size:
        raise SummaryServiceError("SUMMARY_CHUNK_OVERLAP은 청크 크기보다 작아야 합니다.")

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(encoding.decode(tokens[start : start + chunk_size]))
        if start + chunk_size >= len(tokens):
            break
    return chunks


async def _request_structured_response(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_content: str,
    response_model: type[ModelT],
) -> ModelT:
    response = await client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=response_model,
        temperature=0.2,
    )

    if not response.choices:
        raise SummaryServiceError("모델 응답이 비어 있습니다.")

    message = response.choices[0].message
    if message.parsed is not None:
        return message.parsed
    return _parse_result(message.content, response_model)


async def _analyze_chunks(
    client: AsyncOpenAI,
    model_name: str,
    chunks: Sequence[str],
) -> list[ChunkAnalysis]:
    max_concurrency = _positive_int_setting(
        "SUMMARY_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_chunk(index: int, chunk: str) -> ChunkAnalysis:
        async with semaphore:
            content = f"회의록 구간 {index + 1}/{len(chunks)}:\n\n{chunk}"
            return await _request_structured_response(
                client,
                model_name,
                CHUNK_SYSTEM_PROMPT,
                content,
                ChunkAnalysis,
            )

    return await asyncio.gather(
        *(analyze_chunk(index, chunk) for index, chunk in enumerate(chunks))
    )


async def analyze_meeting(text: str) -> AnalyzeResponse:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")

    try:
        client = AsyncOpenAI(api_key=api_key)
        chunks = _chunk_text(text, model_name)
        if not chunks:
            raise SummaryServiceError("회의 전문이 비어 있습니다.")

        chunk_results = await _analyze_chunks(client, model_name, chunks)
        merge_content = json.dumps(
            [result.model_dump(mode="json") for result in chunk_results],
            ensure_ascii=False,
        )
        return await _request_structured_response(
            client,
            model_name,
            MERGE_SYSTEM_PROMPT,
            merge_content,
            AnalyzeResponse,
        )

    except OpenAIError as exc:
        raise SummaryServiceError("OpenAI API 요청에 실패했습니다.") from exc
