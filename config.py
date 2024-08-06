import pydantic
import os
import yaml
from pydantic_settings import BaseSettings
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


def load_config():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')

    try:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return Config(**config_dict)
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
