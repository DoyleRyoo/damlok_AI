from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="분석할 회의 전문 텍스트")


class TextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="처리된 텍스트")


class SttProviderDebugResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = Field(description="해당 provider 실행 성공 여부")
    elapsed_ms: int | None = Field(default=None, description="처리 시간(ms)")
    text: str | None = Field(default=None, description="STT 결과 텍스트")
    error: str | None = Field(default=None, description="실패 사유")


class SttDebugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local: SttProviderDebugResult
    openai: SttProviderDebugResult


class PreprocessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="기존 호환용 필드. cleaned_transcript와 동일")
    raw_transcript: str = Field(description="STT 원문 또는 입력 원문")
    cleaned_transcript: str = Field(description="분석/요약용 basic 전처리 결과")
    readable_transcript: str | None = Field(
        default=None,
        description="사용자 보기용 light_ai 전처리 결과",
    )
    readable_available: bool = Field(description="readable_transcript 생성 성공 여부")
    readable_error: str | None = Field(
        default=None,
        description="readable_transcript 생성 실패 사유",
    )


class TaggingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="기존 호환용 필드. tagged_transcript와 동일")
    raw_transcript: str = Field(description="입력 원문")
    cleaned_transcript: str = Field(description="basic cleanup 결과")
    tagged_transcript: str = Field(description="ACTION/DECISION 등 태그가 붙은 회의 전문")


class PreprocessDebugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = Field(description="적용된 전처리 모드")
    raw_transcript: str = Field(default="", description="입력 원문")
    cleaned_transcript: str = Field(default="", description="basic cleanup 결과")
    input_chars: int = Field(description="입력 텍스트 길이")
    cleaned_chars: int = Field(default=0, description="basic cleanup 결과 길이")
    removed_chars: int = Field(default=0, description="cleanup으로 줄어든 문자 수")
    basic_chars: int = Field(description="기본 전처리 결과 길이")
    ai_chars: int | None = Field(default=None, description="AI 전처리 결과 길이")
    raw_ai_chars: int | None = Field(default=None, description="fallback 적용 전 AI 결과 길이")
    final_processed_chars: int = Field(description="fallback 적용 후 최종 전처리 결과 길이")
    total_chunks: int = Field(default=0, description="AI 전처리 청크 수")
    ai_used_chunks: int = Field(default=0, description="AI 결과를 사용한 청크 수")
    fallback_chunks: int = Field(default=0, description="basic 결과로 fallback된 청크 수")
    too_short_count: int = Field(default=0, description="AI 결과가 너무 짧아 fallback된 청크 수")
    too_long_count: int = Field(default=0, description="AI 결과가 너무 길어 fallback된 청크 수")
    preserved_dates: bool = Field(default=True, description="날짜 정보 보존 여부")
    preserved_numbers: bool = Field(default=True, description="숫자 정보 보존 여부")
    preserved_entities: bool = Field(default=True, description="사람/기관명 정보 보존 여부")
    preserved_directives: bool = Field(default=True, description="지시 표현 보존 여부")
    date_preserve_ratio: float = Field(default=1.0, description="날짜 정보 보존율")
    number_preserve_ratio: float = Field(default=1.0, description="숫자 정보 보존율")
    entity_preserve_ratio: float = Field(default=1.0, description="사람/기관명 정보 보존율")
    directive_preserve_ratio: float = Field(default=1.0, description="지시 표현 보존율")
    section_counts: dict[str, int] = Field(
        default_factory=dict,
        description="섹션 태그별 문단 수",
    )
    action_candidate_count: int = Field(default=0, description="액션후보 문단 수")
    decision_candidate_count: int = Field(default=0, description="결정사항 문단 수")
    ignored_section_count: int = Field(default=0, description="요약 우선 대상에서 제외된 문단 수")
    fallback_reason_by_chunk: list[str | None] = Field(
        default_factory=list,
        description="청크별 fallback 사유",
    )
    fallback_used: bool = Field(default=False, description="AI 전처리 fallback 적용 여부")
    fallback_reason: str | None = Field(default=None, description="AI 전처리 fallback 사유")
    basic_preview: str = Field(description="기본 전처리 결과 미리보기")
    ai_preview: str | None = Field(default=None, description="AI 전처리 결과 미리보기")


class TaggingDebugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_transcript: str = Field(description="입력 원문")
    cleaned_transcript: str = Field(description="basic cleanup 결과")
    tagged_transcript: str = Field(description="ACTION/DECISION 등 태그가 붙은 회의 전문")
    tag_counts: dict[str, int] = Field(description="전체 태그별 문단 수")
    summary_used_tag_counts: dict[str, int] = Field(
        description="summary/full에서 우선 참고하는 태그별 문단 수"
    )
    ignored_tag_counts: dict[str, int] = Field(description="summary/full에서 제외되는 태그별 문단 수")
    action_count: int = Field(description="ACTION 태그 문단 수")
    decision_count: int = Field(description="DECISION 태그 문단 수")
    discussion_count: int = Field(description="DISCUSSION 태그 문단 수")
    chitchat_count: int = Field(description="CHITCHAT 태그 문단 수")


class SttJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class SttJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Supabase Storage object path")
    bucket: str | None = Field(
        default=None,
        description="Supabase Storage bucket name. 기본값은 meeting-audio",
    )


class SttJobCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: SttJobStatus


class SttJobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: SttJobStatus
    text: str | None = None
    error: str | None = None


class ShortSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meeting_summary: str = Field(description="벡터 저장용 회의 한 줄 요약")


class SummaryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(description="회의 목적")
    discussion: str = Field(description="핵심 논의 내용")
    decision: str = Field(description="주요 결정 사항")


class ActionItemPriority(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ActionItemStatus(StrEnum):
    NOT_STARTED = "미착수"
    IN_PROGRESS = "진행중"
    COMPLETED = "완료"


class ActionItemSourceType(StrEnum):
    DIRECTIVE = "DIRECTIVE"
    PLAN = "PLAN"
    SUGGESTION = "SUGGESTION"
    DISCUSSION = "DISCUSSION"


class ActionItemConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ActionItemData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignee_name: str | None = Field(description="담당자 이름")
    assignee_email: str | None = Field(description="담당자 이메일")
    task: str = Field(description="해야 할 일")
    start_date: date | None = Field(description="명시된 시작일")
    due_date: date | None = Field(description="명시된 마감일")
    priority: ActionItemPriority
    status: ActionItemStatus

    @field_validator(
        "assignee_name",
        "assignee_email",
        "start_date",
        "due_date",
        mode="before",
    )
    @classmethod
    def convert_null_like_strings(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "확인되지 않음",
            "미정",
        }:
            return None
        return value


class AiReviewActionItemData(ActionItemData):
    source_type: ActionItemSourceType = Field(
        description="AI 응답/프론트 검토용 임시 메타데이터. DB 저장용 아님",
    )
    confidence: ActionItemConfidence = Field(
        description="AI 응답/프론트 검토용 임시 메타데이터. DB 저장용 아님",
    )
    evidence: str = Field(
        description="AI 응답/프론트 검토용 원문 근거. DB 저장용 아님",
    )
    duplicate_group_id: str | None = Field(
        default=None,
        description="중복 의심 항목을 묶는 프론트 검토용 임시 값. DB 저장용 아님",
    )


class ChunkAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: SummaryData
    action_items: list[AiReviewActionItemData]
    action_candidates: list[AiReviewActionItemData] = Field(
        description="정책 방향/검토 주제 등 사용자 확인 후 등록할 후보"
    )


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: SummaryData
    action_items: list[AiReviewActionItemData]
    action_candidates: list[AiReviewActionItemData] = Field(
        description="AI 응답/프론트 검토용 후보. 확정 전 DB 저장용 아님",
    )
    meeting_summary: str = Field(description="벡터 저장용 회의 한 줄 요약")


class AnalyzeDebugResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_count: int = Field(description="분석에 사용된 청크 수")
    selected_summary_text: str = Field(description="요약 분석에 실제 사용된 텍스트")
    chunks: list[ChunkAnalysis] = Field(description="청크별 중간 분석 결과")
    result: AnalyzeResponse = Field(description="최종 병합 분석 결과")
