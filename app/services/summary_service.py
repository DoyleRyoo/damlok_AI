import asyncio
import json
import os
import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError
import tiktoken

from app.schemas import AnalyzeDebugResponse, AnalyzeResponse, ChunkAnalysis


CHUNK_SYSTEM_PROMPT = """
당신은 회의록 분석 전문가입니다.

입력은 태깅된 회의록을 나눈 일부 구간입니다.
제공된 구간만 근거로 상세 요약, action_items, action_candidates를 추출하세요.
추측하지 말고, 텍스트에서 확인할 수 없는 내용은 null 또는 "확인되지 않음"으로 작성하세요.
JSON null이 필요한 필드에는 문자열 "null"을 절대 쓰지 마세요.
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
      "status": "미착수",
      "source_type": "DIRECTIVE 또는 PLAN 또는 SUGGESTION 또는 DISCUSSION",
      "confidence": "HIGH 또는 MEDIUM 또는 LOW",
      "evidence": "원문 근거 문장",
      "duplicate_group_id": "중복 의심 그룹 id 또는 null"
    }
  ],
  "action_candidates": [
    {
      "assignee_name": "담당자 이름 또는 null",
      "assignee_email": "담당자 이메일 또는 null",
      "task": "검토 후보",
      "start_date": "YYYY-MM-DD 또는 null",
      "due_date": "YYYY-MM-DD 또는 null",
      "priority": "HIGH 또는 MEDIUM 또는 LOW",
      "status": "미착수",
      "source_type": "DIRECTIVE 또는 PLAN 또는 SUGGESTION 또는 DISCUSSION",
      "confidence": "HIGH 또는 MEDIUM 또는 LOW",
      "evidence": "원문 근거 문장",
      "duplicate_group_id": "중복 의심 그룹 id 또는 null"
    }
  ]
}

규칙:
- 이 구간에서 확인되는 내용만 구체적으로 작성하세요.
- objective에는 회의의 배경과 목적을, discussion에는 주요 쟁점과 의견을,
  decision에는 확정된 결정과 미결 사항을 구분해 작성하세요.
- tagged_transcript 입력에서는 DISCUSSION, INFO, QUESTION, ANSWER를 discussion에 참고하세요.
- tagged_transcript 입력에서는 DECISION 태그를 decision에 우선 참고하되, 명확한 결정/확정/합의/보류가 없는 내용을 결정으로 만들지 마세요.
- 회의 진행, 발표 순서, 시간 관계상 발표 생략, 다음 순서 안내 같은 운영성 결정은 summary.decision에 넣지 마세요.
- action_items가 없으면 빈 배열 [] 로 작성하세요.
- action_candidates가 없으면 빈 배열 [] 로 작성하세요.
- action_items에는 ACTION 태그 중 실제 실행 지시성이 높은 항목을 넣으세요. 담당자가 없어도 명확한 지시나 계획이면 assignee_name을 null로 두고 포함할 수 있습니다.
- ACTION 태그라도 검토, 제안, 정책 방향, 개선 필요성에 가까우면 action_items가 아니라 action_candidates에 넣으세요.
- action_candidates에는 정책 방향, 검토 주제, 개선 필요성, 담당자가 불분명하지만 실행 가능성이 있는 항목을 넣으세요.
- QUESTION/ANSWER 태그 안에도 실행 가능한 제안, 검토 답변, 추진 의사가 있으면 action_candidates 근거로 참고하세요.
- action_items와 action_candidates는 관리자/프론트 검토용 AI 응답입니다. source_type, confidence, evidence, duplicate_group_id는 DB 저장용이 아닙니다.
- source_type은 DIRECTIVE, PLAN, SUGGESTION, DISCUSSION 중 하나만 사용하세요.
- confidence는 HIGH, MEDIUM, LOW 중 하나만 사용하세요. HIGH는 실제 Task 저장 가능, MEDIUM은 후보 검토 필요, LOW는 단순 논의에 가깝다는 뜻입니다.
- evidence에는 원문 근거 문장을 짧게 넣으세요.
- 태그가 없는 입력에서는 발언에서 실행 후보가 드러난 항목을 추출하세요.
- 담당자가 명확하지 않으면 assignee_name을 null로 작성하고, 이메일이 명확하지 않으면 assignee_email을 null로 작성하세요.
- 날짜가 명확하지 않거나 없으면 start_date와 due_date를 null로 작성하세요.
- 연도, 월, 일이 모두 명확한 날짜만 YYYY-MM-DD로 작성하세요.
- 연도가 확인되지 않은 "이번 주 금요일", "3월 21일" 같은 표현은 추정하지 말고 null로 작성하세요.
- 단, "오늘", "내일", "모레"처럼 현재 날짜 기준으로 날짜가 하나로 확정되는 표현은 입력에 제공된 현재 날짜 기준으로 계산할 수 있습니다.
- 회의일, 월초, 월말, 분기말, 연말, 사업 종료일, 과거 문맥을 근거로 날짜를 보정하거나 새로 만들지 마세요.
- assignee_name, assignee_email, start_date, due_date에는 문자열 "null"을 쓰지 말고 JSON null을 사용하세요.
- 비슷해 보인다는 이유로 action_items나 action_candidates를 임의 병합하거나 제거하지 마세요. 대상과 행동이 사실상 같은 경우만 중복으로 판단하고, 제거하지 말고 duplicate_group_id로 묶으세요.
- priority는 HIGH, MEDIUM, LOW 중 하나만 사용하세요.
- 완료 또는 진행 중이라고 명시되지 않은 액션 아이템의 status는 "미착수"로 작성하세요.
"""

MERGE_SYSTEM_PROMPT = """
당신은 여러 회의록 구간의 분석 결과를 하나로 통합하는 회의록 분석 전문가입니다.

입력으로 제공된 구간별 분석만 근거로 최종 결과를 작성하세요.
서로 다른 내용, action_items, action_candidates를 누락하지 마세요.
추측하거나 새로운 담당자, 날짜, 결정 사항을 추가하지 마세요.

규칙:
- summary는 회의 내용을 다시 확인하지 않아도 될 만큼 구체적으로 작성하세요.
- objective에는 전체 회의의 배경과 목적을 작성하세요.
- discussion에는 주요 쟁점, 의견, 근거와 미결 사항을 종합하세요.
- decision에는 실제로 확정된 결정 사항만 작성하세요. 명확한 결정/확정/합의/보류가 없으면 논의나 후보를 결정으로 만들지 마세요.
- 회의 진행, 발표 순서, 시간 관계상 발표 생략, 다음 순서 안내 같은 운영성 결정은 decision에 넣지 마세요.
- DISCUSSION, INFO, QUESTION, ANSWER는 discussion에 참고하고, DECISION은 decision에 우선 반영하세요.
- action_items에는 실제 실행 지시성이 높은 항목을 유지하세요.
- action_candidates에는 정책 방향, 검토 주제, 개선 필요성, 담당자가 불분명하지만 실행 가능성이 있는 항목을 유지하세요.
- QUESTION/ANSWER 태그에서 나온 제안, 검토 답변, 추진 의사도 실행 가능성이 있으면 action_candidates로 유지하세요.
- action_items와 action_candidates는 관리자/프론트 검토용 AI 응답입니다. source_type, confidence, evidence, duplicate_group_id는 DB 저장용이 아닙니다.
- 비슷해 보인다는 이유로 action_items나 action_candidates를 임의 병합하거나 제거하지 마세요. 대상과 행동이 사실상 같은 경우만 중복으로 판단하고, 제거하지 말고 duplicate_group_id로 묶으세요.
- 담당자나 날짜가 확인되지 않은 값은 null을 유지하세요.
- 연도, 월, 일이 모두 명확한 날짜만 YYYY-MM-DD로 작성하세요.
- 연도가 확인되지 않은 날짜 표현은 추정하지 말고 null을 유지하세요.
- 단, "오늘", "내일", "모레"처럼 현재 날짜 기준으로 날짜가 하나로 확정되는 표현은 입력에 제공된 현재 날짜 기준으로 계산할 수 있습니다.
- 회의일, 월초, 월말, 분기말, 연말, 사업 종료일, 과거 문맥을 근거로 날짜를 보정하거나 새로 만들지 마세요.
- assignee_name, assignee_email, start_date, due_date에는 문자열 "null"을 쓰지 말고 JSON null을 사용하세요.
- meeting_summary는 상세 요약을 대신하지 않으며, 벡터 검색용 1문장으로 짧게 작성하세요.
"""

ModelT = TypeVar("ModelT", bound=BaseModel)
DEFAULT_CHUNK_SIZE = 6_000
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_MAX_CONCURRENCY = 3
SECTION_TAG_PATTERN = re.compile(
    r"^\s*(\[의전/잡담\]|\[배경설명\]|\[핵심논의\]|\[결정사항\]|\[액션후보\]|\[질문\]|\[응답\]|\[보류/추후검토\]|ACTION:|DECISION:|DISCUSSION:|QUESTION:|ANSWER:|INFO:|CHITCHAT:)"
)
LINE_SECTION_TAG_PATTERN = re.compile(
    r"(?=^\s*(?:\[의전/잡담\]|\[배경설명\]|\[핵심논의\]|\[결정사항\]|\[액션후보\]|\[질문\]|\[응답\]|\[보류/추후검토\]|ACTION:|DECISION:|DISCUSSION:|QUESTION:|ANSWER:|INFO:|CHITCHAT:))",
    re.MULTILINE,
)
SUMMARY_PRIORITY_TAGS = {
    "[핵심논의]",
    "[결정사항]",
    "[액션후보]",
    "ACTION:",
    "DECISION:",
    "DISCUSSION:",
    "QUESTION:",
    "ANSWER:",
    "INFO:",
}
FULL_DATE_PATTERN = re.compile(
    r"(?P<year>\d{4})\s*(?:-|/|\.|년)\s*0?(?P<month>\d{1,2})\s*(?:-|/|\.|월)\s*0?(?P<day>\d{1,2})\s*일?"
)
RELATIVE_DATE_OFFSETS = {
    "오늘": 0,
    "내일": 1,
    "모레": 2,
}
NO_EXPLICIT_DECISION_TEXT = "명확한 결정사항 없음"
DECISION_TAGS = {"[결정사항]", "[보류/추후검토]", "DECISION:"}
OPERATIONAL_DECISION_PATTERNS = (
    "시간 관계상",
    "발표를 진행",
    "발표는 어렵",
    "발표 주제",
    "장관님께서 지정",
    "다음 순서",
    "회의를 마치",
    "진행하도록",
    "발언",
    "토론을 진행",
)
WEAK_ACTION_PATTERNS = (
    "검토",
    "모색",
    "필요",
    "노력",
    "준비",
    "방안",
    "제안",
    "고민",
)


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


def select_summary_sections(text: str) -> str:
    priority_blocks = []
    has_section_tags = False
    normalized_text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    for block in LINE_SECTION_TAG_PATTERN.split(normalized_text):
        stripped = block.strip()
        if not stripped:
            continue
        match = SECTION_TAG_PATTERN.match(stripped)
        if not match:
            continue
        has_section_tags = True
        if match.group(1) in SUMMARY_PRIORITY_TAGS:
            priority_blocks.append(stripped)

    if priority_blocks:
        return "\n\n".join(priority_blocks)
    if has_section_tags:
        return text
    return text


def _current_date_text() -> str:
    return date.today().isoformat()


def _date_evidence_variants(value: date) -> set[str]:
    month = value.month
    day = value.day
    return {
        value.isoformat(),
        f"{value.year}.{month}.{day}",
        f"{value.year}.{month:02d}.{day:02d}",
        f"{value.year}/{month}/{day}",
        f"{value.year}/{month:02d}/{day:02d}",
        f"{value.year}년 {month}월 {day}일",
        f"{value.year}년 {month:02d}월 {day:02d}일",
    }


def _dates_supported_by_text(text: str, current_date: date) -> set[date]:
    supported = set()
    for match in FULL_DATE_PATTERN.finditer(text):
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        try:
            supported.add(date(year, month, day))
        except ValueError:
            continue
    for word, offset in RELATIVE_DATE_OFFSETS.items():
        if word in text:
            supported.add(current_date + timedelta(days=offset))
    return supported


def _date_has_evidence(value: date, evidence_text: str, source_text: str, current_date: date) -> bool:
    combined_text = f"{evidence_text}\n{source_text}"
    if value in _dates_supported_by_text(combined_text, current_date):
        return True
    return any(variant in combined_text for variant in _date_evidence_variants(value))


def _strip_unsupported_dates(result: AnalyzeResponse, source_text: str) -> AnalyzeResponse:
    current_date = date.fromisoformat(_current_date_text())
    payload = result.model_dump(mode="python")
    for group_name in ("action_items", "action_candidates"):
        for item in payload[group_name]:
            evidence = item.get("evidence") or ""
            for field_name in ("start_date", "due_date"):
                value = item.get(field_name)
                if value is None:
                    continue
                if not _date_has_evidence(value, evidence, source_text, current_date):
                    item[field_name] = None
    return AnalyzeResponse.model_validate(payload)


def _tagged_text_has_any_section(text: str) -> bool:
    normalized_text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return any(
        SECTION_TAG_PATTERN.match(block.strip())
        for block in LINE_SECTION_TAG_PATTERN.split(normalized_text)
    )


def _is_operational_decision_block(text: str) -> bool:
    return any(pattern in text for pattern in OPERATIONAL_DECISION_PATTERNS)


def _tagged_text_has_business_decision(text: str) -> bool:
    normalized_text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    for block in LINE_SECTION_TAG_PATTERN.split(normalized_text):
        stripped = block.strip()
        match = SECTION_TAG_PATTERN.match(stripped)
        if match and match.group(1) in DECISION_TAGS:
            if _is_operational_decision_block(stripped):
                continue
            return True
    return False


def _is_candidate_like_action(item: dict[str, object]) -> bool:
    text = f"{item.get('task') or ''}\n{item.get('evidence') or ''}"
    return any(pattern in text for pattern in WEAK_ACTION_PATTERNS)


def _enforce_summary_policy(result: AnalyzeResponse, source_text: str) -> AnalyzeResponse:
    payload = result.model_dump(mode="python")
    if _tagged_text_has_any_section(source_text) and not _tagged_text_has_business_decision(source_text):
        payload["summary"]["decision"] = NO_EXPLICIT_DECISION_TEXT

    strict_items = []
    moved_candidates = []
    for item in payload["action_items"]:
        if _is_candidate_like_action(item):
            candidate = {**item}
            candidate["priority"] = "MEDIUM"
            candidate["source_type"] = "SUGGESTION"
            candidate["confidence"] = "MEDIUM"
            moved_candidates.append(candidate)
            continue
        strict_items.append(item)

    payload["action_items"] = strict_items
    payload["action_candidates"] = payload["action_candidates"] + moved_candidates
    return AnalyzeResponse.model_validate(payload)


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
            content = (
                f"현재 날짜: {_current_date_text()}\n"
                f"회의록 구간 {index + 1}/{len(chunks)}:\n\n{chunk}"
            )
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


async def _analyze_meeting_parts(
    text: str,
) -> tuple[str, list[ChunkAnalysis], AnalyzeResponse]:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")

    try:
        client = AsyncOpenAI(api_key=api_key)
        summary_text = select_summary_sections(text)
        chunks = _chunk_text(summary_text, model_name)
        if not chunks:
            raise SummaryServiceError("회의 전문이 비어 있습니다.")

        chunk_results = await _analyze_chunks(client, model_name, chunks)
        merge_content = json.dumps(
            {
                "current_date": _current_date_text(),
                "chunk_results": [
                    result.model_dump(mode="json") for result in chunk_results
                ],
            },
            ensure_ascii=False,
        )
        result = await _request_structured_response(
            client,
            model_name,
            MERGE_SYSTEM_PROMPT,
            merge_content,
            AnalyzeResponse,
        )
        result = _strip_unsupported_dates(result, summary_text)
        result = _enforce_summary_policy(result, summary_text)
        return summary_text, chunk_results, result

    except OpenAIError as exc:
        raise SummaryServiceError("OpenAI API 요청에 실패했습니다.") from exc


async def analyze_meeting(text: str) -> AnalyzeResponse:
    _, _, result = await _analyze_meeting_parts(text)
    return result


async def analyze_meeting_debug(text: str) -> AnalyzeDebugResponse:
    selected_summary_text, chunk_results, result = await _analyze_meeting_parts(text)
    return AnalyzeDebugResponse(
        chunk_count=len(chunk_results),
        selected_summary_text=selected_summary_text,
        chunks=chunk_results,
        result=result,
    )
