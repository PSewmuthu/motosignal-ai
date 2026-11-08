from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class SignalPhase(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    color: Literal["red", "amber", "green", "unknown"]
    direction: str = "all"

    @model_validator(mode="after")
    def valid_window(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                "Signal phase end_seconds must be greater than start_seconds")

        return self


class AnalysisConfig(BaseModel):
    mode: Literal["same_camera", "separate_cameras"] = "same_camera"
    stop_line: list[Point] = Field(
        default_factory=list, min_length=0, max_length=2)
    approach_side: Literal["positive", "negative", "either"] = "either"
    direction: str = "all"
    signal_timeline: list[SignalPhase] = Field(default_factory=list)
    rear_time_offset_seconds: float = 0.0
    min_travel_seconds: float = 0.0
    max_travel_seconds: float = 15.0
    frame_stride: int = Field(default=1, ge=1, le=120)
    min_track_observations: int = Field(default=3, ge=2, le=100)
    max_track_gap_seconds: float = Field(default=1.5, gt=0, le=10)

    @model_validator(mode="after")
    def validate_config(self):
        if len(self.stop_line) not in (0, 2):
            raise ValueError(
                "stop_line must contain exactly two normalized points or be empty")

        if self.max_travel_seconds < self.min_travel_seconds:
            raise ValueError(
                "max_travel_seconds must be >= min_travel_seconds")

        return self


class CreateJobRequest(BaseModel):
    traffic_video_id: str
    rear_video_id: str | None = None
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def validate_camera_mode(self):
        if self.config.mode == "separate_cameras" and not self.rear_video_id:
            raise ValueError(
                "rear_video_id is required in separate_cameras mode")

        return self
