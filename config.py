import os
import yaml
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComparisonMode(BaseModel):
    enabled: bool
    offer_models: Optional[List[str]]


class Langchain(BaseModel):
    tracing_v2: str  # important that it's not bool
    project: str
    endpoint: str


class TallyForm(BaseModel):
    voting_id: str
    voting_width: int
    voting_height: int
    feedback_id: str
    feedback_width: int
    feedback_height: int


class DateRange(BaseModel):
    default_start: Optional[str]
    default_end: Optional[str]
    min_date: str


class MiscDisplayOptions(BaseModel):
    display_source_texts: bool
    display_topic_data: bool
    display_issue_selector: bool


class LLMModel(BaseModel):
    name: str
    provider: str
    type: str
    temperature: float


class LLM(BaseModel):
    default_model: str
    models: List[LLMModel]


class Config(BaseSettings):
    llm: LLM
    tasks: Dict[str, Dict[str, Any]]
    project_indexes: Dict[str, List[str]]
    comparison_mode: ComparisonMode
    tally_form: TallyForm
    date_range: DateRange
    langchain: Langchain
    misc_display_options: MiscDisplayOptions
    max_doc_num: int
    num_candidates: int

    model_config = SettingsConfigDict(env_file=None, extra='ignore')


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')

    try:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return Config.model_validate(config_dict)
    except FileNotFoundError:
        print(
            f"Config file not found at {config_path}. Please ensure config.yaml is in the same directory as config.py.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing the config file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the config: {e}")
        raise


config = load_config()
