import pandas as pd
import streamlit as st
import plotly.express as px
import logging

from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO)


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
# def populate_default_values(index_name, es_config):
#     """
#     Retrieves unique values for specified fields from an Elasticsearch index
#     and appends an "Any" option to each list from the specified Elasticsearch index.
#     """
#     if "dem-arm" in index_name:
#         category_level_one_values = get_unique_category_values(index_name, 'misc.category_one.keyword', es_config)
#         category_level_two_values = get_unique_category_values(index_name, 'misc.category_two.keyword', es_config)
#         category_level_one_values.append("Any")
#         category_level_two_values.append("Any")
#     elif "ru-balkans" in index_name:
#         category_level_one_values = get_unique_category_values(index_name, 'misc.category_one.keyword', es_config)
#         category_level_one_values.append("Any")
#         category_level_two_values = []
#     else:
#         category_level_one_values = get_unique_category_values(index_name, 'category.keyword', es_config)
#         category_level_one_values.append("Any")
#         category_level_two_values = []
#
#     language_values = get_unique_category_values(index_name, 'language.keyword', es_config)
#     country_values = get_unique_category_values(index_name, 'country.keyword', es_config)
#
#     language_values.append("Any")
#     country_values.append("Any")
#
#     return sorted(category_level_one_values), sorted(category_level_two_values), sorted(language_values), sorted(
#         country_values)
def populate_default_values(index_name, es_config):
    """
    Retrieves unique values for specified fields from an Elasticsearch index
    and appends an "Any" option to each list from the specified Elasticsearch index.
    """
    if "dem-arm" in index_name:
        fields = ['misc.category_one.keyword', 'misc.category_two.keyword', 'language.keyword', 'country.keyword']
    elif "ru-balkans" in index_name:
        fields = ['misc.category_one.keyword', 'language.keyword', 'country.keyword']
    else:
        fields = ['category.keyword', 'language.keyword', 'country.keyword']

    unique_values = get_multiple_unique_values(index_name, fields, es_config)

    category_level_one_values = unique_values.get('misc.category_one.keyword',
                                                  unique_values.get('category.keyword', []))
    category_level_two_values = unique_values.get('misc.category_two.keyword', [])
    language_values = unique_values.get('language.keyword', [])
    country_values = unique_values.get('country.keyword', [])

    # Append "Any" to each list
    category_level_one_values.append("Any")
    if category_level_two_values:
        category_level_two_values.append("Any")
    language_values.append("Any")
    country_values.append("Any")

    return (sorted(category_level_one_values),
            sorted(category_level_two_values),
            sorted(language_values),
            sorted(country_values))


# index_options = [
#     'detector-media-tiktok',
#     'ua-by-facebook',
#     'ua-by-telegram',
#     'ua-by-web',
#     'ua-by-youtube',
#     'dm-8-countries-twitter',
#     'dm-8-countries-telegram',
#     'ndi-lithuania-instagram',
#     'ndi-lithuania-web',
#     'ndi-lithuania-youtube',
#     'ndi-lithuania-telegram',
#     'ndi-lithuania-initial-kivu-twitter',
#     'recovery-win-facebook',
#     'recovery-win-telegram',
#     'recovery-win-web',
#     'recovery-win-twitter',
#     'recovery-win-comments-telegram']

project_indexes = {
    # 'ua-by': [
    #     'ua-by-facebook',
    #     'ua-by-telegram',
    #     'ua-by-web',
    #     'ua-by-youtube'
    # ],
    # 'oip': [
    #     'oip-rt-web'
    # ],
    # 'dem-by': [
    #     'dem-by-instagram',
    #     'dem-by-telegram',
    #     'dem-by-web',
    #     'dem-by-youtube',
    #     'dem-by-vkontakte',
    #     'dem-by-odnoklassniki',
    #     'dem-by-whisper-tiktok'
    # ],
    'dem-arm': [
        'dem-arm-facebook',
        'dem-arm-telegram',
        'dem-arm-web',
        'dem-arm-youtube'
    ]
    # 'dm-8-countries': [
    #     'dm-8-countries-twitter',
    #     'dm-8-countries-telegram'
    # ],
    # 'recovery-win': [
    #     'recovery-win-facebook',
    #     'recovery-win-telegram',
    #     'recovery-win-web',
    #     'recovery-win-twitter',
    #     'recovery-win-comments-telegram'
    # ],
    # 'ru-balkans': [
    #     'ru-balkans-facebook',
    #     'ru-balkans-telegram',
    #     'ru-balkans-youtube',
    #     'ru-balkans-sample-facebook',
    #     'ru-balkans-sample-telegram',
    #     'ru-balkans-sample-youtube'
    # ],
    # 'arabic_test': [
    #     'arabic-translation-test-web',
    #     'cs-disininfo-iq-telegram'
    # ],
    # 'detector-media': [
    #     'detector-media-tiktok',
    # ],
    # 'eu4ge': [
    #     'eu4ge-facebook',
    #     'eu4ge-web',
    #     'eu4ge-instagram',
    #     'eu4ge-telegram',
    #     'eu4ge-tiktok'
}
flat_index_list = [index for indexes in project_indexes.values() for index in indexes]


@st.cache_data(ttl=3600)  # Cache for 1 hour
# def get_prefixed_fields(index_, prefix, es_config):
#     es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
#                        request_timeout=600)
#     base_index = '-'.join(index_.split('-')[:2])
#     indices = es.cat.indices(index=f"{base_index}*", h="index").split()
#
#     all_fields = set()
#
#     for index in indices:
#         mapping = es.indices.get_mapping(index=index)
#         fields = extract_fields(mapping[index]['mappings'], 'issues.')
#         prefixed_fields = [field for field in fields if field.startswith(prefix)]
#         all_fields.update(prefixed_fields)
#
#     return list(all_fields)
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


# def extract_fields(mapping, target_prefix):
#     fields = []
#     if 'properties' in mapping:
#         for field, props in mapping['properties'].items():
#             if field == target_prefix.rstrip('.'):
#                 sub_fields = extract_fields(props, field)
#                 fields += [f"{field}.{sub}" for sub in sub_fields]
#             else:
#                 fields.append(field)
#                 fields += extract_fields(props, "")
#     return fields
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
    Creates a list of 'term' queries for Elasticsearch based on selected items.
    Returns:
        list: A list of 'term' queries for inclusion in an Elasticsearch 'should' clause.
    """
    if (selected_items is None) or ("Any" in selected_items):
        return []
    else:
        return [{"term": {field: item}} for item in selected_items]


# def add_terms_condition(must_list, terms):
#     """
#     Adds individual term to Elasticsearch query.
#     """
#     if terms:
#         must_list.append({
#             "bool": {
#                 "should": terms,
#                 "minimum_should_match": 1
#             }
#         })

def add_terms_condition(must_list, terms, field):
    if terms:
        must_list.append({
            "bool": {
                "must": [{"term": {field: item}} for item in terms]
            }
        })


# def create_must_term(category_one_terms, category_two_terms, language_terms, country_terms, formatted_start_date,
#                      formatted_end_date, thresholds_dict=None):
#     """
#     Constructs a 'must' term for an Elasticsearch query that incorporates
#     filters for date range, category, language, and country.
#
#     Each filter term is added to the 'must' term only if it is not None.
#     """
#
#     must_term = [
#         {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
#     ]
#
#     add_terms_condition(must_term, category_one_terms)
#     add_terms_condition(must_term, category_two_terms)
#     add_terms_condition(must_term, language_terms)
#     add_terms_condition(must_term, country_terms)
#
#     if thresholds_dict:
#         add_issues_conditions(must_term, thresholds_dict)
#
#     return must_term

def create_must_term(category_one_terms, category_two_terms, language_terms, country_terms, formatted_start_date,
                     formatted_end_date, thresholds_dict=None):
    must_term = [
        {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
    ]

    add_terms_condition(must_term, category_one_terms, 'misc.category_one.keyword')
    add_terms_condition(must_term, category_two_terms, 'misc.category_two.keyword')
    add_terms_condition(must_term, language_terms, 'language.keyword')
    add_terms_condition(must_term, country_terms, 'country.keyword')

    if thresholds_dict:
        add_issues_conditions(must_term, thresholds_dict)

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
                'language': doc['_source'].get('language', ''),
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
