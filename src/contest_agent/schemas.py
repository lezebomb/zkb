from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ImagePayload(FlexibleModel):
    format: str = Field(default="url")
    data: str


class InferRequest(FlexibleModel):
    request_id: str
    session_id: str
    task_type: str
    image: ImagePayload
    meta: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(FlexibleModel):
    status: str
    supported_tasks: list[str]
    service: str | None = None
    version: str | None = None
    bridge_mode: str | None = None


class InferResponse(FlexibleModel):
    request_id: str
    task_type: str
    ok: bool
    result: dict[str, Any] | None
    elapsed_ms: int
    message: str
