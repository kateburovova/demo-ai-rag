# Base
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# External
import pandas as pd
import plotly.express as px
import streamlit as st
from angle_emb import AnglE, Prompts
from elasticsearch import BadRequestError, Elasticsearch
from elasticsearch.exceptions import NotFoundError
from langchain import callbacks, hub
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Internal
from config import Config, LLMModel

logging.basicConfig(level=logging.INFO)


def load_es_config() -> Dict[str, str]:
    """
    Load Elasticsearch configuration from Streamlit secrets.

    Args:
    None

    Returns:
    dict: A dictionary containing Elasticsearch configuration with the following keys:
        - 'host': The Elasticsearch host address
        - 'port': The Elasticsearch port number
        - 'api_key': The API key for authentication with Elasticsearch
    """
    es_config = {
        'host': st.secrets['ld_rag']['ELASTIC_HOST'],
        'port': st.secrets['ld_rag']['ELASTIC_PORT'],
        'api_key': st.secrets['ld_rag']['ELASTIC_API']
    }
    return es_config


def get_default_date_range(config: Config) -> Tuple[date, date, date, date]:
    """
    Determine the default date range based on config settings.

    Args:
    config (Config): The configuration object containing date range settings.

    Returns:
    Tuple[date, date, date, date]: A tuple containing four date objects:
        - min_date: The minimum allowed date
        - default_start_date: The default start date for the range
        - default_end_date: The default end date for the range
        - today: The current date
    """
    min_date = datetime.strptime(config.date_range.min_date, '%Y-%m-%d').date()
    today = datetime.now().date()

    if config.date_range.default_end and config.date_range.default_start:
        default_end_date = datetime.strptime(config.date_range.default_end, '%Y-%m-%d').date()
        default_start_date = datetime.strptime(config.date_range.default_start, '%Y-%m-%d').date()
    else:
        default_end_date = today
        default_start_date = today - timedelta(days=13)  # Last 14 days including today

    # Ensure the default dates are not before the min_date
    default_start_date = max(default_start_date, min_date)
    default_end_date = max(default_end_date, min_date)

    return min_date, default_start_date, default_end_date, today


def get_keys() -> Dict[str, str]:
    """
    Retrieve API keys for OpenAI and Anthropic from Streamlit secrets.

    Returns:
    Dict[str, str]: A dictionary containing API keys

    Raises:
    Exception: If there's an error loading the API keys
    """
    try:
        api_keys = {
            'openai': st.secrets['ld_rag']['OPENAI_KEY_ORG'],
            'anthropic': st.secrets['ld_rag']['ANTHROPIC_API_KEY']
        }
        logging.info('Loaded api keys successfully.')
        return api_keys
    except Exception as e:
        logging.error(f'Could not load api keys due to error: {e}')
        return {}


def initialize_llm(model_config: LLMModel, api_keys: Dict[str, str]):
    """
    Initialize a language model based on the provided configuration.

    Args:
    model_config (LLMModel): Configuration for the language model
    api_keys (Dict[str, str]): Dictionary containing API keys for different providers

    Returns:
    BaseChatModel: An initialized language model

    Raises:
    ValueError: If an unsupported model type or provider is specified
    """
    provider = model_config.provider
    model_type = model_config.type
    model_name = model_config.name
    temperature = model_config.temperature

    if provider == 'openai' and model_type == 'ChatOpenAI':
        return ChatOpenAI(
            temperature=temperature,
            openai_api_key=api_keys['openai'],
            model_name=model_name
        )
    elif provider == 'anthropic' and model_type == 'ChatAnthropic':
        return ChatAnthropic(
            temperature=temperature,
            anthropic_api_key=api_keys['anthropic'],
            model_name=model_name
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type} for provider: {provider}")


def init_llms(config: Config, api_keys: Dict[str, str]) -> Dict[str, ChatOpenAI | ChatAnthropic]:
    """
    Initialize multiple language models based on the provided configuration.

    Args:
    config (Config): The configuration object containing LLM settings
    api_keys (Dict[str, str]): Dictionary containing API keys for different providers

    Returns:
    Dict[str, ChatOpenAI | ChatAnthropic]: A dictionary of initialized language models,
        where keys are model names and values are initialized model instances

    Raises:
    ValueError: If an unsupported model type or provider is specified
    Exception: For any other errors during model initialization
    """
    llm_models = {}
    for model in config.llm.models:
        try:
            llm_models[model.name] = initialize_llm(model, api_keys)
        except ValueError as e:
            st.warning(f"Could not initialize model {model.name}: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while initializing model {model.name}: {str(e)}")
    return llm_models


def init_langsmith_params(config: Config) -> None:
    """
    This function sets the following environment variables:
    - LANGCHAIN_TRACING_V2
    - LANGCHAIN_PROJECT
    - LANGCHAIN_ENDPOINT
    - LANGCHAIN_API_KEY
    - LANGSMITH_ACC

    Args:
    config (Config): The configuration object containing LangSmith settings
    """
    os.environ["LANGCHAIN_TRACING_V2"] = config.langchain.tracing_v2
    os.environ["LANGCHAIN_PROJECT"] = config.langchain.project
    os.environ["LANGCHAIN_ENDPOINT"] = config.langchain.endpoint
    os.environ["LANGCHAIN_API_KEY"] = st.secrets['ld_rag']['LANGCHAIN_API_KEY']
    os.environ["LANGSMITH_ACC"] = st.secrets['ld_rag']['LANGSMITH_ACC']


def set_state_defaults() -> None:
    """
    Set default values for various session state variables in a Streamlit application.

    Args:
    None

    Returns:
    None"""

    if 'formatted_start_date' not in st.session_state:
        st.session_state.formatted_start_date = None
    if 'formatted_end_date' not in st.session_state:
        st.session_state.formatted_end_date = None
    if 'thresholds_dict' not in st.session_state:
        st.session_state.thresholds_dict = {}
    if 'show_issues_form' not in st.session_state:
        st.session_state.show_issues_form = False
    if 'category_terms_one' not in st.session_state:
        st.session_state.category_terms_one = None
    if 'category_terms_two' not in st.session_state:
        st.session_state.category_terms_two = None
    if 'language_terms' not in st.session_state:
        st.session_state.language_terms = None
    if 'country_terms' not in st.session_state:
        st.session_state.country_terms = None
    if 'use_base_category' not in st.session_state:
        st.session_state.use_base_category = False
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = None
    if 'compare_categories' not in st.session_state:
        st.session_state.compare_categories = False


def pull_prompts(config: Config) -> Dict[str, ChatPromptTemplate]:
    """
    Pull prompt templates for different tasks from LangSmith.
    If a prompt fails to load, a default prompt is created and used instead.

    Args:
    config (Config): The configuration object containing task settings

    Returns:
    Dict[str, ChatPromptTemplate]: A dictionary where keys are task names and values are
    corresponding ChatPromptTemplate objects

    Raises:
    Exception: If there's an error pulling a prompt from LangSmith
    """
    langsmith_acc = os.environ.get("LANGSMITH_ACC", "")
    logging.info(f'langsmith_acc: {langsmith_acc}')
    prompts = {}

    for task, task_config in config.tasks.items():
        prompt_id = task_config['primary']
        full_prompt_id = f'{langsmith_acc}/{prompt_id}'
        logging.info(f'full_prompt_id: {full_prompt_id}')

        try:
            prompt_template = hub.pull(full_prompt_id)
            prompts[task] = prompt_template
            logging.info(f"Successfully pulled prompt for task: {task}")

        except Exception as e:
            # Create a fallback prompt
            logging.warning(f"Failed to pull prompt for task {task}: {e}")
            system_template = "You are a helpful assistant that summarizes information."
            human_template = "Please summarize the following information:\n\n{texts}\n\nQuestion: {question}"
            default_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
            prompts[task] = default_prompt
            logging.info(f"Using default prompt for task: {task}")

    return prompts


# (prompt_template, llm, texts, placeholder, input_question):

def generate_output_stream(prompt_template: ChatPromptTemplate,
                           llm: Any,
                           texts: List[Any],
                           placeholder: st.empty,
                           input_question: str) -> Tuple[str, str]:
    """
    Generate a stream of output from a language model and display it in a Streamlit app.

    Args:
    prompt_template (ChatPromptTemplate): The template for formatting the prompt
    llm: The language model to use for generation
    texts (List[Any]): The list of texts to be included in the prompt
    placeholder (st.empty): A Streamlit empty placeholder for displaying the output
    input_question (str): The input question to be answered

    Returns:
    Tuple[str, str]: A tuple containing:
        - The generated content as a string
        - The run ID of the generation process
    """
    customer_messages = prompt_template.format_messages(
        question=input_question,
        texts=texts
    )
    with callbacks.collect_runs() as cb:
        content = []
        for chunk in llm.stream(customer_messages):
            content.append(chunk.content)
            placeholder.markdown("".join(content))
        run_id = cb.traced_runs[0].id
    return "".join(content), run_id


def get_unique_category_values(index_name: str, field: str, es_config: Dict[str, str]) -> List[str]:
    """
    Retrieve unique values from a specified field in an Elasticsearch index.

    Args:
    index_name (str): The name of the Elasticsearch index to search
    field (str): The field name to aggregate unique values from
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key

    Returns:
    List[str]: A list of unique values from the specified field

    Raises:
    Exception: If there's an error retrieving values from Elasticsearch
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                           request_timeout=300)

        agg_query = {
            "size": 0,
            "aggs": {
                "unique_categories": {
                    "terms": {"field": field, "size": 10000}
                }
            }
        }

        response = es.search(index=index_name, body=agg_query)
        unique_values = [bucket['key'] for bucket in response['aggregations']['unique_categories']['buckets']]

        return unique_values
    except Exception as e:
        logging.error(f"Error retrieving unique values from {field}: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_multiple_unique_values(index_name: str, fields: List[str], es_config: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Retrieve unique values from multiple fields in a specified Elasticsearch index.
    It is cached using Streamlit's st.cache_data decorator with a respective TTL.

    Args:
    index_name (str): The name of the Elasticsearch index to search
    fields (List[str]): A list of field names to aggregate unique values from
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key

    Returns:
    Dict[str, List[str]]: A dictionary with field names as keys and lists of unique values as values

    Raises:
    Exception: If there's an error retrieving values from Elasticsearch
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                           request_timeout=300)

        agg_query = {
            "size": 0,
            "aggs": {
                f"unique_{field.replace('.', '_')}": {
                    "terms": {"field": field, "size": 10000}
                } for field in fields
            }
        }

        response = es.search(index=index_name, body=agg_query)

        result = {}
        for field in fields:
            agg_key = f"unique_{field.replace('.', '_')}"
            result[field] = [bucket['key'] for bucket in response['aggregations'][agg_key]['buckets']]

        return result
    except Exception as e:
        logging.error(f"Error retrieving unique values: {e}")
        return {field: [] for field in fields}


@st.cache_data(ttl=3600, show_spinner=False)
def populate_default_values(index_name: str, es_config: Dict[str, str]) -> Tuple[
    List[str], List[str], List[str], List[str]]:
    """
    Retrieves unique values for specified fields from an Elasticsearch index
    and appends an "Any" option to each list.

    Args:
    index_name (str): The name of the Elasticsearch index to search
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key

    Returns:
    Tuple[List[str], List[str], List[str], List[str]]: A tuple containing four lists:
        - category_level_one_values: Unique values for category level one
        - category_level_two_values: Unique values for category level two (if applicable)
        - language_values: Unique values for languages
        - country_values: Unique values for countries

    Note:
    This function is cached using Streamlit's st.cache_data decorator with a TTL of 1 hour.
    The fields queried depend on the index name (e.g., 'dem-arm', 'ru-balkans', or others).
    An "Any" option is appended to each list of unique values.
    """

    logging.info(f"Populating selectors for index name: {index_name}")
    if "dem-arm" in index_name:
        logging.info(f"Populating for categories with dem-arm: {index_name}")
        fields = ['misc.category_one.keyword', 'misc.category_two.keyword', 'language_text.keyword', 'country.keyword']
    elif "ru-balkans" in index_name:
        logging.info(f"Populating for categories with ru-balkans: {index_name}")
        fields = ['misc.category_one.keyword', 'language_text.keyword', 'country.keyword']
    else:
        logging.info(f"Populating for base categories: {index_name}")
        fields = ['category.keyword', 'language_text.keyword', 'country.keyword']

    unique_values = get_multiple_unique_values(index_name, fields, es_config)

    category_level_one_values = unique_values.get('misc.category_one.keyword',
                                                  unique_values.get('category.keyword', []))
    category_level_two_values = unique_values.get('misc.category_two.keyword', [])
    language_values = unique_values.get('language_text.keyword', [])
    country_values = unique_values.get('country.keyword', [])

    # Append "Any" to each list
    category_level_one_values.append("Any")
    if category_level_two_values:
        category_level_two_values.append("Any")
    language_values.append("Any")
    country_values.append("Any")

    logging.info(f"Unique values: {unique_values}")

    return (sorted(category_level_one_values),
            sorted(category_level_two_values),
            sorted(language_values),
            sorted(country_values))


def get_category_field() -> str:
    """
    Determine the appropriate category field based on the selected index.
    This function relies on the 'selected_index' value in the Streamlit session state.

    Returns:
    str: The appropriate category field name
        - 'misc.category_one.keyword' for 'dem-arm' or 'ru-balkans' indexes
        - 'category.keyword' for all other indexes
    """
    if "dem-arm" in st.session_state.selected_index or "ru-balkans" in st.session_state.selected_index:
        return 'misc.category_one.keyword'
    else:
        return 'category.keyword'


def get_texts_from_elastic(
        input_question: str,
        question_vector: List[float],
        must_term: List[Dict[str, Any]],
        es_config: Dict[str, str],
        config: Config) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """
    Retrieve relevant texts from Elasticsearch based on the input question and filters.
    Correct URLs to ensure they start with 'http://' or 'https://'.

    Args:
    input_question (str): The user's input question
    question_vector (List[float]): The vector representation of the input question
    must_term (List[Dict[str, Any]]): A list of filter conditions for the Elasticsearch query
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key
    config (Config): The configuration object containing search settings

    Returns:
    Tuple[List[Tuple[str, str, str]], Dict[str, Any]]: A tuple containing:
        - A list of tuples, each containing (text, url, category) for relevant documents
        - The raw Elasticsearch response

    Raises:
    BadRequestError: If there's an error executing the search (e.g., missing embeddings)
    NotFoundError: If the specified index is not found
    Exception: For any other errors during the search process
    """
    try:
        texts_list = []
        st.write(f'Running search for {config.max_doc_num} relevant posts for question: {input_question}')
        try:
            es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                               request_timeout=600)
        except Exception as e:
            st.error(f'Failed to connect to Elasticsearch: {str(e)}')

        with st.spinner("Searching for relevant documents, please wait ..."):
            response = es.search(index=st.session_state.selected_index,
                                 size=config.max_doc_num,
                                 knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                      "query_vector": question_vector,
                                      "k": config.max_doc_num,
                                      "num_candidates": config.num_candidates,
                                      "filter": {
                                          "bool": {
                                              "must": must_term,
                                              "must_not": [{"term": {"type": "comment"}}]
                                          }
                                      }
                                      }
                                 )

        logging.info(f"Total hits: {response['hits']['total']['value']}")
        logging.info(
            f"Sample document: {response['hits']['hits'][0] if response['hits']['hits'] else 'No hits'}")

        category_field = get_category_field()
        logging.info(f'Category_field: {category_field}')

        for doc in response['hits']['hits']:
            if st.session_state.use_base_category:
                logging.info(f'st.session_state.use_base_category : {st.session_state.use_base_category}')
                category = doc['_source'].get('category', None)
            else:
                if '.' in category_field:
                    parts = category_field.split('.')
                    logging.info(f'Part {parts[0], parts[1]}')
                    category = doc['_source'].get(parts[0], {}).get(parts[1], 'Unknown')
                else:
                    category = doc['_source'].get(category_field, 'Unknown')

            texts_list.append((doc['_source']['translated_text'], doc['_source']['url'], category))

        # Format urls so they work properly within streamlit
        corrected_texts_list = [(text, 'https://' + url if not url.startswith('http://') and not url.startswith(
            'https://') else url, category) for text, url, category in texts_list]

        return corrected_texts_list, response

    except BadRequestError as e:
        st.error(f'Failed to execute search (embeddings might be missing for this index): {e.info}')
        return [], None
    except NotFoundError as e:
        st.error(f'Index not found: {e.info}')
        return [], None
    except Exception as e:
        st.error(f'An unknown error occurred: {str(e)}')
        return [], None


@st.cache_resource(hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None}, show_spinner=False)
def load_model() -> AnglE:
    """
    Load and initialize the AnglE embedding model - the 'WhereIsAI/UAE-Large-V1' model with 'cls' pooling strategy.
    The cache is set up to ignore thread locks and weak references for hashing.

    Returns:
    AnglE: An initialized AnglE model instance
    """
    angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
                                        pooling_strategy='cls')
    return angle_model


def get_guestion_vector(input_question: str) -> List[float]:
    """
    Generate a vector representation of the input question using the AnglE model.
    The encoding is done using the 'C' prompt from the AnglE.Prompts enum.

    Args:
    input_question (str): The input question to be vectorized

    Returns:
    List[float]: A list of floats representing the vector encoding of the input question
    """
    angle = load_model()
    vec = angle.encode({'text': input_question}, to_numpy=True, prompt=Prompts.C)
    question_vector = vec.tolist()[0]
    return question_vector


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_prefixed_fields(index_: str, prefix: str, es_config: Dict[str, str]) -> List[str]:
    """
    Retrieve field names with a specific prefix from Elasticsearch index mappings.
    Searches for fields in all indices matching the base index pattern derived from the input index name.

    Args:
    index_ (str): The name of the Elasticsearch index (or index pattern)
    prefix (str): The prefix to filter field names
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key

    Returns:
    List[str]: A list of field names that start with the given prefix
    """
    es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                       request_timeout=600)
    base_index = '-'.join(index_.split('-')[:2])

    # Get all indices matching the base index pattern
    indices = es.cat.indices(index=f"{base_index}*", h="index", format="json")
    index_names = [index['index'] for index in indices]

    # Get mappings for all matching indices in one request
    mappings = es.indices.get_mapping(index=",".join(index_names))

    all_fields = set()

    for index, index_mapping in mappings.items():
        fields = extract_fields(index_mapping['mappings'], prefix)
        all_fields.update(fields)

    return list(all_fields)


def add_issues_conditions(must_list: List[Dict], thresholds_dict: Dict[str, str]) -> None:
    """
    Adds "issues" field conditions to the Elasticsearch query based on a given dictionary of thresholds.

    Args:
    must_list (List[Dict]): The list of must conditions for the Elasticsearch query
    thresholds_dict (Dict[str, str]): A dictionary with keys as "issues" fields and values as threshold ranges
    in the format "min:max"

    Returns:
    None

    Note:
    This function modifies the must_list in-place by appending a new condition for each issue field.
    The condition uses Elasticsearch's range query to filter documents based on the provided thresholds.
    """
    issues_conditions = []

    for issue_field, threshold in thresholds_dict.items():
        min_value, max_value = map(float, threshold.split(":"))

        issues_conditions.append({
            "range": {
                issue_field: {
                    "gte": min_value,
                    "lte": max_value
                }
            }
        })

    must_list.append({
        "bool": {
            "should": issues_conditions,
            "minimum_should_match": 1
        }
    })


def extract_fields(mapping: Dict[str, Any], prefix: str, current_path: str = '') -> List[str]:
    """
    Recursively extract field names from an Elasticsearch mapping that start with a given prefix.
    Builds the field names by concatenating the current path with field names in the mapping.

    Args:
    mapping (Dict[str, Any]): The Elasticsearch mapping dictionary
    prefix (str): The prefix to filter field names
    current_path (str): The current path in the mapping hierarchy (default: '')

    Returns:
    List[str]: A list of field names that start with the given prefix
    """
    fields = []
    if 'properties' in mapping:
        for field, props in mapping['properties'].items():
            new_path = f"{current_path}.{field}" if current_path else field
            if new_path.startswith(prefix):
                fields.append(new_path)
            fields.extend(extract_fields(props, prefix, new_path))
    return fields


def populate_terms(selected_items: Optional[List[str]], field: str) -> List[str]:
    """
    Creates a list of terms for Elasticsearch based on selected items.
    This function filters out the "Any" option and returns an empty list if "Any" is selected
    or if selected_items is None.

    Args:
    selected_items (Optional[List[str]]): A list of selected items or None
    field (str): The field name associated with the selected items

    Returns:
    List[str]: A list of selected terms, excluding "Any" and empty when None or "Any" is selected
    """
    logging.info(f"Populating terms for {field}: {selected_items}")
    if (selected_items is None) or ("Any" in selected_items):
        return []
    else:
        return selected_items


def add_terms_condition(must_list: List[Dict], terms: List[str], field: str) -> None:
    """
    Adds a terms condition to the Elasticsearch query if terms are provided.

    Args:
    must_list (List[Dict]): The list of must conditions for the Elasticsearch query
    terms (List[str]): The list of terms to be added to the condition
    field (str): The field name to which the terms condition should be applied

    Returns:
    None
    """
    if terms:
        must_list.append({
            "terms": {field: terms}
        })


def create_must_term(
        category_one_terms: List[str],
        category_two_terms: List[str],
        language_terms: List[str],
        country_terms: List[str],
        formatted_start_date: str,
        formatted_end_date: str,
        thresholds_dict: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    Create a list of must terms for an Elasticsearch query based on various filters.
    This function constructs filters based on category, language, country, date range, and thresholds.
    It uses session state to determine whether to use base categories or misc categories.

    Args:
    category_one_terms (List[str]): Terms for the first category level
    category_two_terms (List[str]): Terms for the second category level
    language_terms (List[str]): Terms for language filtering
    country_terms (List[str]): Terms for country filtering
    formatted_start_date (str): Start date for the date range filter
    formatted_end_date (str): End date for the date range filter
    thresholds_dict (Optional[Dict[str, str]]): Dictionary of threshold values for issues filtering

    Returns:
    List[Dict]: A list of must terms to be used in an Elasticsearch query
    """
    logging.info(f"At the start of create_must_term, use_base_category: {st.session_state.use_base_category}")

    logging.info(
        f"Creating must term with: category_one_terms={category_one_terms}, category_two_terms={category_two_terms}")
    must_term = [
        {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
    ]

    if category_one_terms:
        if st.session_state.use_base_category:
            add_terms_condition(must_term, category_one_terms, 'category.keyword')
            logging.info(f"Using base category. After adding category: {must_term}")
        else:
            add_terms_condition(must_term, category_one_terms, 'misc.category_one.keyword')
            logging.info(f"After adding category one: {must_term}")

    if category_two_terms:
        add_terms_condition(must_term, category_two_terms, 'misc.category_two.keyword')
        logging.info(f"After adding category two: {must_term}")
    else:
        logging.info("Category two terms were empty, not added to query")

    if language_terms:
        add_terms_condition(must_term, language_terms, 'language_text.keyword')
    if country_terms:
        add_terms_condition(must_term, country_terms, 'country.keyword')

    if thresholds_dict:
        add_issues_conditions(must_term, thresholds_dict)

    logging.info(f"Final must term: {must_term}")
    return must_term


def create_dataframe_from_response(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from Elasticsearch response data.

    Args:
    response (Dict[str, Any]): The Elasticsearch response containing search hits

    Returns:
    pd.DataFrame: A DataFrame containing selected fields from the response
    """
    try:
        selected_documents = []

        if 'hits' not in response or 'hits' not in response['hits']:
            print("No data found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame

        for doc in response['hits']['hits']:
            misc_dict = doc['_source'].get('misc', {})

            selected_doc = {
                'date': doc['_source'].get('date', 'None'),
                'text': doc['_source'].get('text', 'None'),
                'url': doc['_source'].get('url', 'None'),
                'country': doc['_source'].get('country', 'None'),
                'language': doc['_source'].get('language_text', 'None'),
                'category': doc['_source'].get('category', 'None'),
                'source': doc['_source'].get('source', 'None'),
                '_domain': doc['_source'].get('_domain', 'None'),
                'category_one': misc_dict.get('category_one', 'None'),
                'category_two': misc_dict.get('category_two', 'None'),
                'id': doc.get('_id', 'None')
            }
            selected_documents.append(selected_doc)

        df_selected_fields = pd.DataFrame(selected_documents)

        if 'date' in df_selected_fields.columns:
            df_selected_fields['date'] = pd.to_datetime(df_selected_fields['date']).dt.date

        return df_selected_fields

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def display_distribution_charts(df: pd.DataFrame, selected_index: str) -> None:
    """
    Displays donut charts for category, language, and country distributions in a Streamlit application.

    Args:
    df (pd.DataFrame): The DataFrame containing the data to be visualized
    selected_index (str): The name of the selected Elasticsearch index

    Returns:
    None

    Note:
    This function creates different chart layouts based on the selected index:
    - For 'dem-arm' indexes, it displays four charts: category one, category two, language, and country
    - For other indexes, it displays three charts: category, language, and country
    The charts are created using Plotly Express and displayed in Streamlit columns.
    If the DataFrame is empty, a message is displayed instead of charts.
    """

    if df.empty:
        st.write("No data available to display.")
        return

    if 'dem-arm' in selected_index:
        col1, col2, col3, col4 = st.columns(4)

        if 'category_one' in df.columns:
            category_counts = df['category_one'].value_counts().reset_index()
            category_counts.columns = ['category_one', 'count']
            fig_category = px.pie(category_counts, names='category_one', values='count',
                                  title='Category One Distribution', hole=0.4)
            col1.plotly_chart(fig_category, use_container_width=True)

        if 'category_two' in df.columns:
            category_counts = df['category_two'].value_counts().reset_index()
            category_counts.columns = ['category_two', 'count']
            fig_category = px.pie(category_counts, names='category_two', values='count',
                                  title='Category Two Distribution', hole=0.4)
            col2.plotly_chart(fig_category, use_container_width=True)

        if 'language' in df.columns:
            language_counts = df['language'].value_counts().reset_index()
            language_counts.columns = ['language', 'count']
            fig_language = px.pie(language_counts, names='language', values='count',
                                  title='Language Distribution', hole=0.4)
            col3.plotly_chart(fig_language, use_container_width=True)

        if 'country' in df.columns:
            country_counts = df['country'].value_counts().reset_index()
            country_counts.columns = ['country', 'count']
            fig_country = px.pie(country_counts, names='country', values='count',
                                 title='Country Distribution', hole=0.4)
            col4.plotly_chart(fig_country, use_container_width=True)

    else:
        col1, col2, col3 = st.columns(3)

        if 'category' in df.columns:
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            fig_category = px.pie(category_counts, names='category', values='count',
                                  title='Category Distribution', hole=0.4)
            col1.plotly_chart(fig_category, use_container_width=True)

        if 'language' in df.columns:
            language_counts = df['language'].value_counts().reset_index()
            language_counts.columns = ['language', 'count']
            fig_language = px.pie(language_counts, names='language', values='count',
                                  title='Language Distribution', hole=0.4)
            col2.plotly_chart(fig_language, use_container_width=True)

        if 'country' in df.columns:
            country_counts = df['country'].value_counts().reset_index()
            country_counts.columns = ['country', 'count']
            fig_country = px.pie(country_counts, names='country', values='count',
                                 title='Country Distribution', hole=0.4)
            col3.plotly_chart(fig_country, use_container_width=True)


# Not used currently
def create_dataframe_from_response_filtered(response: Dict[str, Any], score_threshold: float = 0.7) -> pd.DataFrame:
    """
    Creates a filtered pandas DataFrame from Elasticsearch response data based on a similarity score threshold.

    Args:
    response (Dict[str, Any]): The Elasticsearch response containing search hits
    score_threshold (float): The minimum similarity score for including a hit in the DataFrame (default: 0.7)

    Returns:
    pd.DataFrame: A DataFrame containing data from hits that meet or exceed the score threshold

    Note:
    This function filters the Elasticsearch hits based on the '_score' field.
    It includes all fields from the '_source' of each hit that meets the threshold.
    The similarity score is added as an additional column 'similarity_score' in the resulting DataFrame.
    """
    records = []
    for hit in response['hits']['hits']:
        if hit['_score'] >= score_threshold:
            source = hit['_source']
            similarity_score = hit['_score']
            source['similarity_score'] = similarity_score
            records.append(source)

    df = pd.DataFrame(records)

    return df


# Not used currently
def search_elastic_below_threshold(
        es_config: Dict[str, str],
        selected_index: str,
        question_vector: List[float],
        must_term: List[Dict],
        max_doc_num: int = 10000) -> Optional[pd.DataFrame]:
    """
    Perform an Elasticsearch search with k-NN query and additional filters.

    Args:
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key
    selected_index (str): The name of the Elasticsearch index to search
    question_vector (List[float]): The vector representation of the question for k-NN search
    must_term (List[Dict]): Additional filter conditions for the Elasticsearch query
    max_doc_num (int): Maximum number of documents to retrieve (default: 10000)

    Returns:
    Optional[pd.DataFrame]: A DataFrame containing the search results, or None if an error occurs

    Raises:
    Exception: If there's an error connecting to Elasticsearch or executing the search
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                           request_timeout=600)

        response = es.search(index=selected_index,
                             size=max_doc_num,
                             knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                  "query_vector": question_vector,
                                  "k": 100,
                                  "num_candidates": 10000,
                                  # "similarity": 20, # l2 norm, so not the [0,1]
                                  "filter": {
                                      "bool": {
                                          "must": must_term
                                      }
                                  }
                                  }
                             )
        df = create_dataframe_from_response_filtered(response)
        return df

    except Exception as e:
        st.error(f'Failed to connect to Elasticsearch: {str(e)}')

        return None


def get_topic_counts(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame of topic counts from Elasticsearch response data.

    Args:
    response (Dict[str, Any]): The Elasticsearch response containing search hits

    Returns:
    pd.DataFrame: A DataFrame containing topic IDs and their counts, with columns 'topic_ids' and 'count'

    Note:
    This function extracts topic information from the 'topics' field in each hit's source.
    It creates a list of topic hash IDs for each document, then explodes this list to count occurrences.
    If no topics are found or an error occurs, an empty DataFrame with 'topic_ids' and 'count' columns is returned.
    """
    try:
        selected_documents = []
        for doc in response['hits']['hits']:
            selected_doc = {
                'topic_info': doc['_source'].get('topics', 'None'),
                'id': doc.get('_id', 'None')
            }
            selected_documents.append(selected_doc)
        df_selected_fields = pd.DataFrame(selected_documents)

        df_selected_fields['topic_ids'] = (df_selected_fields.topic_info
                                           .apply(lambda x: [topic['topic_hash_id'] for topic in x
                                                             if isinstance(x, list)
                                                             and isinstance(topic, dict)
                                                             and 'topic_hash_id' in topic]))
        logging.info(f'Retrieved {len(df_selected_fields)} topics.')
        df_topics = df_selected_fields['topic_ids'].explode().value_counts().reset_index()
        return df_topics

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame(columns=['topic_ids', 'count'])


def extract_prefix(index_name: str) -> str:
    """
    Extract the prefix from an Elasticsearch index name by removing platform suffixes.
    Function splits the index name by hyphens and rejoins the parts after removing the suffixes.

    Args:
    index_name (str): The full name of the Elasticsearch index

    Returns:
    str: The extracted prefix of the index name
    """
    platform_suffixes = ['tiktok',
                         'facebook',
                         'twitter',
                         'whisper',
                         'instagram',
                         'telegram',
                         'odnoklassniki',
                         'vkontakte',
                         'web',
                         'youtube',
                         'comments']
    parts = index_name.split('-')
    parts_cleaned = [part for part in parts if part not in platform_suffixes]
    return '-'.join(parts_cleaned)


def infer_topic_index_names(index_string: str) -> str:
    """
    Infer the names of topic indices based on a given string of data index names.

    Args:
    index_string (str): A comma-separated string of data index names

    Returns:
    str: A comma-separated string of inferred topic index names

    Note:
    This function splits the input string into individual index names.
    It extracts the prefix of each index name using the extract_prefix function.
    The topic index names are created by prepending 'topics-' to each extracted prefix.
    Duplicate topic index names are removed.
    """
    indices = index_string.split(',')
    prefixes = [extract_prefix(index) for index in indices]
    topic_indexes = ['topics-' + prefix for prefix in prefixes]
    logging.info(f'Inferred topic indexes: {topic_indexes}')

    return ','.join(list(set(topic_indexes)))


def get_summary_and_narratives(df: pd.DataFrame, index_name: str, es_config: Dict[str, str]) -> pd.DataFrame:
    """
    Add summary and narratives columns to the DataFrame by querying Elasticsearch.

    Args:
    df (pd.DataFrame): Input DataFrame with 'topic_ids' column
    index_name (str): Elasticsearch index name for topic data
    es_config (Dict[str, str]): Elasticsearch configuration containing host, port, and API key

    Returns:
    pd.DataFrame: DataFrame with added 'summary' and 'narratives' columns,
                  rows with both None or empty dropped

    Raises:
    Exception: If there's an error connecting to Elasticsearch
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                           request_timeout=600)
    except Exception as e:
        logging.error(f'Failed to connect to Elasticsearch: {str(e)}')

    def query_es(topic_hash_id):
        query = {
            "query": {
                "term": {
                    "topic_hash_id.keyword": topic_hash_id
                }
            }
        }

        try:
            response = es.search(index=index_name, body=query)

            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]['_source']
                return {
                    'summary': hit.get('topic_summary'),
                    'narratives': hit.get('topic_narratives')
                }
            else:
                return {
                    'summary': None,
                    'narratives': None
                }

        except Exception as e:
            print(f"Error querying Elasticsearch for topic_hash_id {topic_hash_id}: {str(e)}")
            return {
                'summary': None,
                'narratives': None
            }

    results = df['topic_ids'].apply(query_es)
    df['summary'] = results.apply(lambda x: x['summary'])
    df['narratives'] = results.apply(lambda x: x['narratives'])

    def is_empty(value):
        return value is None or value == ""

    df_cleaned = df[~(df['summary'].apply(is_empty) & df['narratives'].apply(is_empty))]

    if len(df_cleaned) < len(df):
        df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned
