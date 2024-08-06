from pydantic import BaseSettings
from typing import Dict, Any, List, Optional


class ComparisonMode(BaseSettings):
    enabled: bool
    offer_models: Optional[List[str]]


class TallyForm(BaseSettings):
    voting_id: str
    voting_width: int
    voting_height: int
    feedback_id: str
    feedback_width: int
    feedback_height: int


class DateRange(BaseSettings):
    default_start: Optional[str]
    default_end: Optional[str]
    min_date: str


class MiscDisplayOptions(BaseSettings):
    display_source_texts: bool
    display_topic_data: bool
    display_issue_selector: bool


class Config(BaseSettings):
    llm: Dict[str, Any]
    tasks: Dict[str, Dict[str, Any]]
    project_indexes: Dict[str, List[str]]
    comparison_mode: ComparisonMode
    tally_form: TallyForm
    date_range: DateRange
    langchain: Dict[str, str]
    misc_display_options: MiscDisplayOptions
    max_doc_num: int
    num_candidates: int

    class Config:
        env_file = "config.yaml"
        env_file_encoding = "yaml"


config = Config()
