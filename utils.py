import pandas as pd
import streamlit as st
import plotly.express as px
import logging
import os

from elasticsearch import Elasticsearch
from langchain import hub, callbacks
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)

def init_llm_params():
    # Init Langchain and Langsmith services
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"rag_app : summarization : production"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets['ld_rag']['LANGCHAIN_API_KEY']
    os.environ["LANGSMITH_ACC"] = st.secrets['ld_rag']['LANGSMITH_ACC']

    # Init openai model
    OPENAI_API_KEY = st.secrets['ld_rag']['OPENAI_KEY_ORG']
    llm_chat = ChatOpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY,
                          model_name='gpt-4-1106-preview')
    return llm_chat

def set_state_defaults():
    must_term = None
    issues_fields = None

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


def get_unique_category_values(index_name, field, es_config):
    """
    Retrieve unique values from the field in a specified Elasticsearch index.
    Returns:
    list: A list of unique values from the 'category.keyword' field.
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


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_multiple_unique_values(index_name, fields, es_config):
    """
    Retrieve unique values from multiple fields in a specified Elasticsearch index.

    Args:
    index_name (str): The name of the Elasticsearch index.
    fields (list): A list of field names to aggregate.
    es_config (dict): Elasticsearch configuration.

    Returns:
    dict: A dictionary with field names as keys and lists of unique values as values.
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


@st.cache_data(ttl=3600)
def populate_default_values(index_name, es_config):
    """
    Retrieves unique values for specified fields from an Elasticsearch index
    and appends an "Any" option to each list from the specified Elasticsearch index.
    """
    logging.info(f"Populating selectors for index name: {index_name}")
    if "dem-arm" in index_name:
        fields = ['misc.category_one.keyword', 'misc.category_two.keyword', 'language_text.keyword', 'country.keyword']
    elif "ru-balkans" in index_name:
        fields = ['misc.category_one.keyword', 'language_text.keyword', 'country.keyword']
    else:
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


project_indexes = {
    'dem-arm': [
        'dem-arm-facebook',
        'dem-arm-telegram',
        'dem-arm-web',
        'dem-arm-youtube'
    ]
}
flat_index_list = [index for indexes in project_indexes.values() for index in indexes]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_prefixed_fields(index_, prefix, es_config):
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


def add_issues_conditions(must_list, thresholds_dict):
    """
    Adds "issues" field conditions to the Elasticsearch query based on a given dictionary of thresholds.
    thresholds_dict: A dictionary with keys as "issues" fields and values as threshold ranges in the format "min:max".
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


def extract_fields(mapping, prefix, current_path=''):
    fields = []
    if 'properties' in mapping:
        for field, props in mapping['properties'].items():
            new_path = f"{current_path}.{field}" if current_path else field
            if new_path.startswith(prefix):
                fields.append(new_path)
            fields.extend(extract_fields(props, prefix, new_path))
    return fields


def populate_terms(selected_items, field):
    """
    Creates a list of terms for Elasticsearch based on selected items.
    Returns:
        list: A list of selected terms.
    """
    logging.info(f"Populating terms for {field}: {selected_items}")
    if (selected_items is None) or ("Any" in selected_items):
        return []
    else:
        return selected_items


def add_terms_condition(must_list, terms, field):
    if terms:
        must_list.append({
            "terms": {field: terms}
        })


def create_must_term(category_one_terms, category_two_terms, language_terms, country_terms, formatted_start_date,
                     formatted_end_date, thresholds_dict=None):
    logging.info(
        f"Creating must term with: category_one_terms={category_one_terms}, category_two_terms={category_two_terms}")
    must_term = [
        {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
    ]

    if category_one_terms:
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


def create_dataframe_from_response(response):
    """
    Creates a pandas DataFrame from Elasticsearch response data.
    Returns:
        pd.DataFrame: A DataFrame containing the selected fields from the response.
    """
    try:
        selected_documents = []

        if 'hits' not in response or 'hits' not in response['hits']:
            print("No data found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame

        for doc in response['hits']['hits']:
            misc_dict = doc['_source'].get('misc', {})

            selected_doc = {
                'date': doc['_source'].get('date', ''),
                'text': doc['_source'].get('text', ''),
                # 'translated_text': doc['_source'].get('translated_text', ''),
                'url': doc['_source'].get('url', ''),
                'country': doc['_source'].get('country', ''),
                'language': doc['_source'].get('language_text', ''),
                'category': doc['_source'].get('category', ''),
                'source': doc['_source'].get('source', ''),
                '_domain': doc['_source'].get('_domain', ''),
                'category_one': misc_dict.get('category_one', ''),
                'category_two': misc_dict.get('category_two', ''),
                'id': doc.get('_id', '')
            }
            selected_documents.append(selected_doc)

        df_selected_fields = pd.DataFrame(selected_documents)

        if 'date' in df_selected_fields.columns:
            df_selected_fields['date'] = pd.to_datetime(df_selected_fields['date']).dt.date

        return df_selected_fields

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def display_distribution_charts(df, selected_index):
    """
    Displays donut charts for category, language, and country distributions in Streamlit.
    The layout is three columns with one donut chart in each column.
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


def create_dataframe_from_response_filtered(response, score_threshold=0.7):
    records = []
    for hit in response['hits']['hits']:
        if hit['_score'] >= score_threshold:
            source = hit['_source']
            similarity_score = hit['_score']
            source['similarity_score'] = similarity_score
            records.append(source)

    df = pd.DataFrame(records)

    return df


def search_elastic_below_threshold(es_config, selected_index, question_vector, must_term, max_doc_num=10000):
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
