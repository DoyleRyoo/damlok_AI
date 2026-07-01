from dataclasses import dataclass
from collections.abc import Sequence
import asyncio
import os
import re

from openai import AsyncOpenAI, OpenAIError
import tiktoken

from app.schemas import (
    PreprocessDebugResponse,
    PreprocessResponse,
    TaggingDebugResponse,
    TaggingResponse,
)
from app.services.summary_service import SummaryServiceError


LIGHT_AI_PREPROCESS_SYSTEM_PROMPT = """
당신은 회의 STT 원문을 섹션 태깅하는 전문가입니다.

입력은 회의 음성에서 변환된 STT 원문 일부입니다.
전처리 목표는 요약이나 문장 정제가 아니라, 원문 문단의 성격을 분류하는 것입니다.

반드시 지킬 규칙:
- 원문 문장 표현, 어순, 조사, 문장부호를 수정하지 마세요.
- 문장을 요약하거나 합치거나 자연스럽게 바꾸지 마세요.
- 원문 내용을 삭제하지 마세요.
- 각 문단 앞에 아래 태그 중 하나만 붙이세요.
  [의전/잡담], [배경설명], [핵심논의], [결정사항], [액션후보], [질문], [응답], [보류/추후검토]
- 이미 태그가 붙은 문단은 적절하면 그대로 유지하세요.
- 출력은 태그가 붙은 회의 전문 텍스트만 작성하세요. 설명, 제목, JSON, 마크다운은 쓰지 마세요.
"""

TAGGING_SYSTEM_PROMPT = """
당신은 회의 STT 원문을 실무 파이프라인용으로 태깅하는 전문가입니다.

입력은 basic cleanup이 끝난 회의 전문 일부입니다.
목표는 요약이나 재작성 없이 문단 또는 발화 단위의 성격을 분류하는 것입니다.

반드시 지킬 규칙:
- 원문 문장 표현, 어순, 조사, 문장부호를 수정하지 마세요.
- 문장을 요약하거나 합치거나 자연스럽게 바꾸지 마세요.
- 원문 내용을 삭제하지 마세요.
- 각 문단 또는 발화 앞에 아래 태그 중 하나만 붙이세요.
  ACTION: 실제 실행 지시 또는 실행 계획
  DECISION: 회의에서 결정, 확정, 합의, 승인, 보류하기로 한 사항
  DISCUSSION: 핵심 논의 내용
  QUESTION: 질문
  ANSWER: 답변
  INFO: 배경 설명, 사례 설명, 참고 정보
  CHITCHAT: 의전, 잡담, 진행 멘트
- 이미 태그가 붙은 문단은 적절하면 그대로 유지하세요.
- ACTION은 넓게 잡으세요. 명확한 지시뿐 아니라 "하겠습니다", "추진하겠습니다", "실시할 예정입니다", "계획입니다", "검토하겠습니다", "공유하겠습니다", "배포하겠습니다", "점검하겠습니다", "감독하겠습니다", "구축하겠습니다", "강화하겠습니다"처럼 실행 가능성이 있는 계획/예정/검토 문장은 ACTION으로 분류하세요.
- "필요합니다", "해야 합니다", "하면 좋겠습니다", "만들어 주셨으면 좋겠습니다"처럼 업무 후보가 될 수 있는 제안이나 개선 필요성도 ACTION으로 분류하세요. 이후 summary/full에서 실제 action_items와 action_candidates로 나눕니다.
- INFO는 사고 개요, 통계, 배경 설명, 현황 보고, 사례 설명, 제도 설명, 참고 정보에만 사용하세요. 앞으로 할 일이나 실행 계획이 드러나면 INFO가 아니라 ACTION으로 분류하세요.
- DECISION은 좁게 잡으세요. "하기로 했습니다", "결정했습니다", "확정했습니다", "합의했습니다", "채택했습니다", "보류하기로 했습니다", "제외하기로 했습니다"처럼 회의 결론이 명확한 문장만 DECISION으로 분류하세요.
- 단순 방향성, 강조 발언, 다짐, 필요성 언급은 DECISION이 아닙니다. 이런 문장은 DISCUSSION 또는 INFO로 분류하세요.
- 질문은 QUESTION, 답변은 ANSWER로 원문 성격을 보존하되, 답변 안에 명확한 실행 계획이 있으면 ACTION으로 분류하세요.
- 출력은 태그가 붙은 회의 전문 텍스트만 작성하세요. 설명, 제목, JSON, 마크다운은 쓰지 마세요.
"""

AGGRESSIVE_AI_PREPROCESS_SYSTEM_PROMPT = """
당신은 회의 STT 원문을 사용자가 읽기 쉽게 다듬는 전문가입니다.

입력은 회의 음성에서 변환된 STT 원문 일부입니다.
요약 전 분석용 전문이 아니라, 사람이 빠르게 읽기 위한 별도 보기용 텍스트로 정리하세요.

반드시 지킬 규칙:
- 핵심 의미, 날짜, 숫자, 사람/기관명, 업무 지시는 보존하세요.
- 불필요한 추임새, 반복, 말더듬, 장황한 연결어는 제거하세요.
- 문단을 읽기 쉽게 정리하고 문장을 자연스럽게 다듬으세요.
- 텍스트에서 확인되지 않은 내용을 새로 만들지 마세요.
- 출력은 정제된 회의 텍스트만 작성하세요. 설명, 제목, JSON, 마크다운은 쓰지 마세요.
"""

DEFAULT_PREPROCESS_CHUNK_SIZE = 5_000
DEFAULT_PREPROCESS_CHUNK_OVERLAP = 0
DEFAULT_PREPROCESS_MAX_CONCURRENCY = 2
DEFAULT_PREPROCESS_DEBUG_PREVIEW_CHARS = 1_000
DEFAULT_PREPROCESS_MIN_LENGTH_RATIO = 0.55
DEFAULT_PREPROCESS_MAX_LENGTH_RATIO = 1.0
DATE_PRESERVE_THRESHOLD = 0.7
NUMBER_PRESERVE_THRESHOLD = 0.7
ENTITY_PRESERVE_THRESHOLD = 0.6
DIRECTIVE_PRESERVE_THRESHOLD = 0.7

DATE_PATTERN = re.compile(
    r"\b\d{4}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2}일?\b|"
    r"\b\d{1,2}\s*월\s*\d{1,2}\s*일\b|"
    r"\b\d{1,2}\s*일\b|"
    r"\b이번\s*주\s*(?:월|화|수|목|금|토|일)요일\b|"
    r"\b다음\s*주\s*(?:월|화|수|목|금|토|일)요일\b|"
    r"\b오늘\b|\b내일\b|\b모레\b"
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[,.]\d+)*(?:\s*(?:명|개|건|곳|억|만|원|%|퍼센트|월|일|년|차|회|분|시간))?\b")
ENTITY_PATTERN = re.compile(
    r"\b[가-힣A-Za-z0-9·]{2,30}(?:고용노동청|노동청|지청|장관|청장|국장|과장|"
    r"위원회|공단|공사|기관|부처|부|처|청|센터|산단|업체|회사|기업|팀|부서|님)\b|"
    r"\b[A-Z][A-Za-z0-9-]{1,20}\b"
)
DIRECTIVE_PATTERN = re.compile(
    r"(?:해\s*주시기\s*바랍니다|해\s*주세요|해야\s*합니다|해야\s*겠습니다|"
    r"필요합니다|검토(?:해)?\s*주시기|점검(?:해)?\s*주시기|추진(?:해)?\s*주시기|"
    r"조치(?:해)?\s*주시기|확인(?:해)?\s*주시기|보고(?:해)?\s*주시기|"
    r"제출(?:해)?\s*주시기|관리(?:해)?\s*주시기)"
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?:[.!?。]\s+|다\.\s+|요\.\s+|\n+)")
ORDER_NUMBER_CONTEXT_PATTERN = re.compile(
    r"(?:첫째|둘째|셋째|넷째|다섯째|여섯째|일곱째|여덟째|아홉째|열째|"
    r"\d+\s*(?:단계|번째|차례|번|순위))"
)
VAGUE_MONTH_CONTEXT_PATTERN = re.compile(r"\d{1,2}\s*월\s*(?:초|중순|말|초순|하순)")
FILLER_ONLY_PATTERN = re.compile(r"^(?:어|음|네|네네|그|그러니까|저기|뭐|아)+[,.!? ]*$")
SECTION_TAGS = (
    "[의전/잡담]",
    "[배경설명]",
    "[핵심논의]",
    "[결정사항]",
    "[액션후보]",
    "[질문]",
    "[응답]",
    "[보류/추후검토]",
)
SUMMARY_PRIORITY_TAGS = {"[핵심논의]", "[결정사항]", "[액션후보]"}
SECTION_TAG_PATTERN = re.compile(
    r"^\s*(\[의전/잡담\]|\[배경설명\]|\[핵심논의\]|\[결정사항\]|\[액션후보\]|\[질문\]|\[응답\]|\[보류/추후검토\])"
)
PIPELINE_TAGS = (
    "ACTION",
    "DECISION",
    "DISCUSSION",
    "QUESTION",
    "ANSWER",
    "INFO",
    "CHITCHAT",
)
PIPELINE_SUMMARY_USED_TAGS = {
    "ACTION",
    "DECISION",
    "DISCUSSION",
    "QUESTION",
    "ANSWER",
    "INFO",
}
PIPELINE_TAG_PATTERN = re.compile(
    r"^\s*(ACTION|DECISION|DISCUSSION|QUESTION|ANSWER|INFO|CHITCHAT):"
)


@dataclass(frozen=True)
class PreprocessChunkResult:
    text: str
    raw_text: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    preserved_dates: bool = True
    preserved_numbers: bool = True
    preserved_entities: bool = True
    preserved_directives: bool = True
    date_preserve_ratio: float = 1.0
    number_preserve_ratio: float = 1.0
    entity_preserve_ratio: float = 1.0
    directive_preserve_ratio: float = 1.0
    date_total: int = 0
    date_preserved: int = 0
    number_total: int = 0
    number_preserved: int = 0
    entity_total: int = 0
    entity_preserved: int = 0
    directive_total: int = 0
    directive_preserved: int = 0


@dataclass(frozen=True)
class PreprocessAiResult:
    text: str
    raw_text: str
    fallback_used: bool
    fallback_reason: str | None
    total_chunks: int
    ai_used_chunks: int
    fallback_chunks: int
    too_short_count: int
    too_long_count: int
    preserved_dates: bool
    preserved_numbers: bool
    preserved_entities: bool
    preserved_directives: bool
    date_preserve_ratio: float
    number_preserve_ratio: float
    entity_preserve_ratio: float
    directive_preserve_ratio: float
    section_counts: dict[str, int]
    action_candidate_count: int
    decision_candidate_count: int
    ignored_section_count: int
    fallback_reason_by_chunk: list[str | None]


def preprocess_transcript(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = []
    previous_line = None
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        line = _remove_consecutive_duplicate_words(line)
        if not line or FILLER_ONLY_PATTERN.fullmatch(line):
            lines.append("")
            continue
        line = _remove_consecutive_duplicate_sentences(line)
        if line == previous_line:
            continue
        lines.append(line)
        previous_line = line
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _remove_consecutive_duplicate_words(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"(?<!\S)(\S+)(?:\s+\1)+(?!\S)", r"\1", current)
    return current


def _remove_consecutive_duplicate_sentences(text: str) -> str:
    parts = re.findall(r"[^.!?。\n]+(?:[.!?。]|$)", text)
    if len(parts) <= 1:
        return text
    deduped = []
    previous = None
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped == previous:
            continue
        deduped.append(stripped)
        previous = stripped
    return " ".join(deduped)


def section_counts(text: str) -> dict[str, int]:
    counts = {tag: 0 for tag in SECTION_TAGS}
    for block in re.split(r"\n{2,}", text):
        match = SECTION_TAG_PATTERN.match(block)
        if match:
            counts[match.group(1)] += 1
    return {tag: count for tag, count in counts.items() if count}


def tagging_counts(text: str) -> dict[str, int]:
    counts = {tag: 0 for tag in PIPELINE_TAGS}
    for block in re.split(r"\n{2,}", text):
        match = PIPELINE_TAG_PATTERN.match(block.strip())
        if match:
            counts[match.group(1)] += 1
    return {tag: count for tag, count in counts.items() if count}


def _ignored_section_count(counts: dict[str, int]) -> int:
    return sum(
        count for tag, count in counts.items() if tag not in SUMMARY_PRIORITY_TAGS
    )


def _normalize_for_presence(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _is_deletion_only(source_text: str, output_text: str) -> bool:
    source_index = 0
    source_length = len(source_text)
    for char in output_text:
        while source_index < source_length and source_text[source_index] != char:
            source_index += 1
        if source_index >= source_length:
            return False
        source_index += 1
    return True


def _extract_unique(pattern: re.Pattern[str], text: str) -> set[str]:
    values = set()
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if value:
            values.add(value)
    return values


def _filter_dates(values: set[str], source_text: str) -> set[str]:
    filtered = set()
    for value in values:
        normalized = _normalize_for_presence(value)
        if normalized.endswith("월") and re.search(
            rf"{re.escape(value)}\s*(?:초|중순|말|초순|하순)",
            source_text,
        ):
            continue
        filtered.add(value)
    return filtered


def _filter_numbers(values: set[str], source_text: str) -> set[str]:
    filtered = set()
    for value in values:
        if not re.search(r"(?:명|개|건|곳|억|만|원|%|퍼센트|월|일|년|차|회|분|시간)", value):
            if re.search(rf"{re.escape(value)}\s*(?:단계|번째|차례|번|순위)", source_text):
                continue
            if ORDER_NUMBER_CONTEXT_PATTERN.search(source_text):
                continue
            continue
        if re.search(rf"{re.escape(value)}\s*(?:초|중순|말|초순|하순)", source_text):
            continue
        filtered.add(value)
    return filtered


def _count_preserved(values: set[str], output_text: str) -> tuple[int, int, float]:
    normalized_output = _normalize_for_presence(output_text)
    total = len(values)
    if total == 0:
        return 0, 0, 1.0
    preserved = sum(
        1 for value in values if _normalize_for_presence(value) in normalized_output
    )
    return preserved, total, preserved / total


def _split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    sentences = re.findall(r"[^.!?。\n]+(?:[.!?。]|다\.|요\.)?", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _critical_directives_preserved(source_text: str, output_text: str) -> bool:
    normalized_output = _normalize_for_presence(output_text)
    for sentence in _split_sentences(source_text):
        dates = _filter_dates(_extract_unique(DATE_PATTERN, sentence), sentence)
        entities = _extract_unique(ENTITY_PATTERN, sentence)
        directives = _extract_unique(DIRECTIVE_PATTERN, sentence)
        if not (dates and entities and directives):
            continue
        values = dates | entities | directives
        if not all(_normalize_for_presence(value) in normalized_output for value in values):
            return False
    return True


def _preservation_checks(source_text: str, output_text: str) -> dict[str, object]:
    dates = _filter_dates(_extract_unique(DATE_PATTERN, source_text), source_text)
    numbers = _filter_numbers(_extract_unique(NUMBER_PATTERN, source_text), source_text)
    entities = _extract_unique(ENTITY_PATTERN, source_text)
    directives = _extract_unique(DIRECTIVE_PATTERN, source_text)

    date_preserved, date_total, date_ratio = _count_preserved(dates, output_text)
    number_preserved, number_total, number_ratio = _count_preserved(numbers, output_text)
    entity_preserved, entity_total, entity_ratio = _count_preserved(entities, output_text)
    directive_preserved, directive_total, directive_ratio = _count_preserved(
        directives,
        output_text,
    )

    critical_directives = _critical_directives_preserved(source_text, output_text)
    return {
        "dates": date_ratio >= DATE_PRESERVE_THRESHOLD,
        "numbers": number_ratio >= NUMBER_PRESERVE_THRESHOLD,
        "entities": entity_ratio >= ENTITY_PRESERVE_THRESHOLD,
        "directives": directive_ratio >= DIRECTIVE_PRESERVE_THRESHOLD
        and critical_directives,
        "date_ratio": date_ratio,
        "number_ratio": number_ratio,
        "entity_ratio": entity_ratio,
        "directive_ratio": directive_ratio,
        "date_total": date_total,
        "date_preserved": date_preserved,
        "number_total": number_total,
        "number_preserved": number_preserved,
        "entity_total": entity_total,
        "entity_preserved": entity_preserved,
        "directive_total": directive_total,
        "directive_preserved": directive_preserved,
        "critical_directives": critical_directives,
    }


def _mode_setting() -> str:
    mode = os.getenv("PREPROCESS_MODE", "light_ai").lower()
    if mode == "ai":
        return "light_ai"
    return mode


def _preview(text: str) -> str:
    preview_chars = _positive_int_setting(
        "PREPROCESS_DEBUG_PREVIEW_CHARS",
        DEFAULT_PREPROCESS_DEBUG_PREVIEW_CHARS,
    )
    if len(text) <= preview_chars:
        return text
    return f"{text[:preview_chars]}..."


def _positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise SummaryServiceError(f"{name} 설정이 올바르지 않습니다.") from exc

    if value <= 0:
        raise SummaryServiceError(f"{name} 설정은 0보다 커야 합니다.")
    return value


def _non_negative_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise SummaryServiceError(f"{name} 설정이 올바르지 않습니다.") from exc

    if value < 0:
        raise SummaryServiceError(f"{name} 설정은 0 이상이어야 합니다.")
    return value


def _ratio_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise SummaryServiceError(f"{name} 설정이 올바르지 않습니다.") from exc

    if value <= 0 or value > 1:
        raise SummaryServiceError(f"{name} 설정은 0보다 크고 1 이하여야 합니다.")
    return value


def _aggregate_ratio(
    chunk_results: Sequence[PreprocessChunkResult],
    preserved_attr: str,
    total_attr: str,
) -> float:
    preserved = sum(getattr(result, preserved_attr) for result in chunk_results)
    total = sum(getattr(result, total_attr) for result in chunk_results)
    if total == 0:
        return 1.0
    return preserved / total


def _chunk_text(text: str, model_name: str) -> list[str]:
    chunk_size = _positive_int_setting(
        "PREPROCESS_CHUNK_TOKENS",
        DEFAULT_PREPROCESS_CHUNK_SIZE,
    )
    overlap = _non_negative_int_setting(
        "PREPROCESS_CHUNK_OVERLAP",
        DEFAULT_PREPROCESS_CHUNK_OVERLAP,
    )
    if overlap >= chunk_size:
        raise SummaryServiceError("PREPROCESS_CHUNK_OVERLAP은 청크 크기보다 작아야 합니다.")

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


async def _preprocess_chunks(
    client: AsyncOpenAI,
    model_name: str,
    chunks: Sequence[str],
    mode: str,
) -> list[PreprocessChunkResult]:
    max_concurrency = _positive_int_setting(
        "PREPROCESS_MAX_CONCURRENCY",
        DEFAULT_PREPROCESS_MAX_CONCURRENCY,
    )
    min_length_ratio = _ratio_setting(
        "PREPROCESS_MIN_LENGTH_RATIO",
        DEFAULT_PREPROCESS_MIN_LENGTH_RATIO,
    )
    max_length_ratio = _ratio_setting(
        "PREPROCESS_MAX_LENGTH_RATIO",
        DEFAULT_PREPROCESS_MAX_LENGTH_RATIO,
    )
    if min_length_ratio > max_length_ratio:
        raise SummaryServiceError(
            "PREPROCESS_MIN_LENGTH_RATIO는 PREPROCESS_MAX_LENGTH_RATIO보다 작거나 같아야 합니다."
        )
    semaphore = asyncio.Semaphore(max_concurrency)
    system_prompt = (
        AGGRESSIVE_AI_PREPROCESS_SYSTEM_PROMPT
        if mode == "aggressive_ai"
        else LIGHT_AI_PREPROCESS_SYSTEM_PROMPT
    )

    async def preprocess_chunk(index: int, chunk: str) -> PreprocessChunkResult:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"회의 STT 구간 {index + 1}/{len(chunks)}:\n\n{chunk}",
                    },
                ],
                temperature=0,
            )
            if not response.choices:
                raise SummaryServiceError("전처리 모델 응답이 비어 있습니다.")
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise SummaryServiceError("전처리 모델 응답이 비어 있습니다.")
            cleaned = content.strip()
            chunk_text = chunk.strip()
            if mode == "light_ai":
                return PreprocessChunkResult(text=cleaned, raw_text=cleaned)

            checks = _preservation_checks(chunk_text, cleaned)
            missing_reasons = []
            thresholds = {
                "dates": DATE_PRESERVE_THRESHOLD,
                "numbers": NUMBER_PRESERVE_THRESHOLD,
                "entities": ENTITY_PRESERVE_THRESHOLD,
                "directives": DIRECTIVE_PRESERVE_THRESHOLD,
            }
            ratio_keys = {
                "dates": "date_ratio",
                "numbers": "number_ratio",
                "entities": "entity_ratio",
                "directives": "directive_ratio",
            }
            for key in ("dates", "numbers", "entities", "directives"):
                if checks[key]:
                    continue
                ratio = float(checks[ratio_keys[key]])
                threshold = thresholds[key]
                if key == "directives" and not checks["critical_directives"]:
                    missing_reasons.append(
                        f"critical directive missing ({ratio:.0%} < {threshold:.0%})"
                    )
                else:
                    missing_reasons.append(f"{key} {ratio:.0%} < {threshold:.0%}")

            common_result = {
                "preserved_dates": bool(checks["dates"]),
                "preserved_numbers": bool(checks["numbers"]),
                "preserved_entities": bool(checks["entities"]),
                "preserved_directives": bool(checks["directives"]),
                "date_preserve_ratio": float(checks["date_ratio"]),
                "number_preserve_ratio": float(checks["number_ratio"]),
                "entity_preserve_ratio": float(checks["entity_ratio"]),
                "directive_preserve_ratio": float(checks["directive_ratio"]),
                "date_total": int(checks["date_total"]),
                "date_preserved": int(checks["date_preserved"]),
                "number_total": int(checks["number_total"]),
                "number_preserved": int(checks["number_preserved"]),
                "entity_total": int(checks["entity_total"]),
                "entity_preserved": int(checks["entity_preserved"]),
                "directive_total": int(checks["directive_total"]),
                "directive_preserved": int(checks["directive_preserved"]),
            }
            if len(cleaned) < len(chunk_text) * min_length_ratio:
                return PreprocessChunkResult(
                    text=chunk_text,
                    raw_text=cleaned,
                    fallback_used=True,
                    fallback_reason="AI output too short",
                    **common_result,
                )
            if len(cleaned) > len(chunk_text) * max_length_ratio:
                return PreprocessChunkResult(
                    text=chunk_text,
                    raw_text=cleaned,
                    fallback_used=True,
                    fallback_reason="AI output too long",
                    **common_result,
                )
            return PreprocessChunkResult(
                text=cleaned,
                raw_text=cleaned,
                **common_result,
            )

    return await asyncio.gather(
        *(preprocess_chunk(index, chunk) for index, chunk in enumerate(chunks))
    )


async def _preprocess_transcript_ai_result(text: str, mode: str = "light_ai") -> PreprocessAiResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")

    model_name = os.getenv("PREPROCESS_MODEL", os.getenv("MODEL_NAME", "gpt-4o-mini"))
    chunks = _chunk_text(text, model_name)
    if not chunks:
        raise SummaryServiceError("전처리할 텍스트가 비어 있습니다.")

    try:
        client = AsyncOpenAI(api_key=api_key)
        chunk_results = await _preprocess_chunks(client, model_name, chunks, mode)
    except OpenAIError as exc:
        raise SummaryServiceError("OpenAI 전처리 요청에 실패했습니다.") from exc

    fallback_reasons = sorted(
        {
            result.fallback_reason
            for result in chunk_results
            if result.fallback_used and result.fallback_reason
        }
    )
    too_short_count = sum(
        result.fallback_reason == "AI output too short" for result in chunk_results
    )
    too_long_count = sum(
        result.fallback_reason == "AI output too long" for result in chunk_results
    )
    fallback_chunks = sum(result.fallback_used for result in chunk_results)
    fallback_reason_by_chunk = [result.fallback_reason for result in chunk_results]
    date_preserve_ratio = _aggregate_ratio(
        chunk_results,
        "date_preserved",
        "date_total",
    )
    number_preserve_ratio = _aggregate_ratio(
        chunk_results,
        "number_preserved",
        "number_total",
    )
    entity_preserve_ratio = _aggregate_ratio(
        chunk_results,
        "entity_preserved",
        "entity_total",
    )
    directive_preserve_ratio = _aggregate_ratio(
        chunk_results,
        "directive_preserved",
        "directive_total",
    )
    counts = section_counts(
        preprocess_transcript("\n\n".join(result.text for result in chunk_results))
    )
    return PreprocessAiResult(
        text=preprocess_transcript("\n\n".join(result.text for result in chunk_results)),
        raw_text=preprocess_transcript(
            "\n\n".join(result.raw_text for result in chunk_results)
        ),
        fallback_used=bool(fallback_reasons),
        fallback_reason=", ".join(fallback_reasons) if fallback_reasons else None,
        total_chunks=len(chunk_results),
        ai_used_chunks=len(chunk_results) - fallback_chunks,
        fallback_chunks=fallback_chunks,
        too_short_count=too_short_count,
        too_long_count=too_long_count,
        preserved_dates=all(result.preserved_dates for result in chunk_results),
        preserved_numbers=all(result.preserved_numbers for result in chunk_results),
        preserved_entities=all(result.preserved_entities for result in chunk_results),
        preserved_directives=all(result.preserved_directives for result in chunk_results),
        date_preserve_ratio=date_preserve_ratio,
        number_preserve_ratio=number_preserve_ratio,
        entity_preserve_ratio=entity_preserve_ratio,
        directive_preserve_ratio=directive_preserve_ratio,
        section_counts=counts,
        action_candidate_count=counts.get("[액션후보]", 0),
        decision_candidate_count=counts.get("[결정사항]", 0),
        ignored_section_count=_ignored_section_count(counts),
        fallback_reason_by_chunk=fallback_reason_by_chunk,
    )


async def preprocess_transcript_ai(text: str, mode: str = "light_ai") -> str:
    result = await _preprocess_transcript_ai_result(text, mode)
    return result.text


async def tag_transcript_ai(text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummaryServiceError("OPENAI_API_KEY가 설정되지 않았습니다.")

    model_name = os.getenv("TAGGING_MODEL", os.getenv("PREPROCESS_MODEL", os.getenv("MODEL_NAME", "gpt-4o-mini")))
    chunks = _chunk_text(text, model_name)
    if not chunks:
        raise SummaryServiceError("태깅할 텍스트가 비어 있습니다.")

    max_concurrency = _positive_int_setting(
        "PREPROCESS_MAX_CONCURRENCY",
        DEFAULT_PREPROCESS_MAX_CONCURRENCY,
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key)

    async def tag_chunk(index: int, chunk: str) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TAGGING_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"회의 STT 구간 {index + 1}/{len(chunks)}:\n\n{chunk}",
                    },
                ],
                temperature=0,
            )
            if not response.choices:
                raise SummaryServiceError("태깅 모델 응답이 비어 있습니다.")
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise SummaryServiceError("태깅 모델 응답이 비어 있습니다.")
            return content.strip()

    try:
        tagged_chunks = await asyncio.gather(
            *(tag_chunk(index, chunk) for index, chunk in enumerate(chunks))
        )
    except OpenAIError as exc:
        raise SummaryServiceError("OpenAI 태깅 요청에 실패했습니다.") from exc

    return preprocess_transcript("\n\n".join(tagged_chunks))


async def tag_meeting_transcript(text: str) -> TaggingResponse:
    cleaned = preprocess_transcript(text)
    tagged = await tag_transcript_ai(cleaned)
    return TaggingResponse(
        text=tagged,
        raw_transcript=text,
        cleaned_transcript=cleaned,
        tagged_transcript=tagged,
    )


async def debug_tag_meeting_transcript(text: str) -> TaggingDebugResponse:
    cleaned = preprocess_transcript(text)
    tagged = await tag_transcript_ai(cleaned)
    counts = tagging_counts(tagged)
    summary_used_counts = {
        tag: count
        for tag, count in counts.items()
        if tag in PIPELINE_SUMMARY_USED_TAGS
    }
    ignored_counts = {
        tag: count
        for tag, count in counts.items()
        if tag not in PIPELINE_SUMMARY_USED_TAGS
    }
    return TaggingDebugResponse(
        raw_transcript=text,
        cleaned_transcript=cleaned,
        tagged_transcript=tagged,
        tag_counts=counts,
        summary_used_tag_counts=summary_used_counts,
        ignored_tag_counts=ignored_counts,
        action_count=counts.get("ACTION", 0),
        decision_count=counts.get("DECISION", 0),
        discussion_count=counts.get("DISCUSSION", 0),
        chitchat_count=counts.get("CHITCHAT", 0),
    )


async def preprocess_meeting_transcript(text: str) -> str:
    cleaned = preprocess_transcript(text)
    mode = _mode_setting()
    if mode == "basic":
        return cleaned
    if mode in {"light_ai", "aggressive_ai"}:
        safe_mode = "light_ai"
        return await preprocess_transcript_ai(cleaned, safe_mode)
    raise SummaryServiceError(
        "PREPROCESS_MODE는 basic, light_ai, aggressive_ai 중 하나여야 합니다."
    )


async def preprocess_meeting_transcripts(text: str) -> PreprocessResponse:
    cleaned = preprocess_transcript(text)
    return PreprocessResponse(
        text=cleaned,
        raw_transcript=text,
        cleaned_transcript=cleaned,
        readable_transcript=None,
        readable_available=False,
        readable_error=None,
    )


async def debug_preprocess_meeting_transcript(text: str) -> PreprocessDebugResponse:
    basic = preprocess_transcript(text)
    return PreprocessDebugResponse(
        mode="basic",
        raw_transcript=text,
        cleaned_transcript=basic,
        input_chars=len(text),
        cleaned_chars=len(basic),
        removed_chars=max(len(text) - len(basic), 0),
        basic_chars=len(basic),
        ai_chars=None,
        raw_ai_chars=None,
        final_processed_chars=len(basic),
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
        basic_preview=_preview(basic),
        ai_preview=None,
    )
