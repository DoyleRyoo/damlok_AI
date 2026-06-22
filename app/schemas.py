from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="분석할 회의 전문 텍스트")


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


class ActionItemData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignee_name: str | None = Field(description="담당자 이름")
    assignee_email: str | None = Field(description="담당자 이메일")
    task: str = Field(description="해야 할 일")
    start_date: date | None = Field(description="명시된 시작일")
    due_date: date | None = Field(description="명시된 마감일")
    priority: ActionItemPriority
    status: ActionItemStatus


class ChunkAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: SummaryData
    action_items: list[ActionItemData]


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: SummaryData
    action_items: list[ActionItemData]
    meeting_summary: str = Field(description="벡터 저장용 회의 한 줄 요약")
