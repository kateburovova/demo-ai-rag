# Base
import os
import time

# Internal
from authentificate import check_password
from utils import (display_distribution_charts, populate_default_values, project_indexes,
                   populate_terms, create_must_term, create_dataframe_from_response, flat_index_list,
                   get_prefixed_fields)

# External
import streamlit as st
import streamlit.components.v1 as components
from langchain import hub, callbacks
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch, BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts


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

es_config = {
    'host': st.secrets['ld_rag']['ELASTIC_HOST'],
    'port': st.secrets['ld_rag']['ELASTIC_PORT'],
    'api_key': st.secrets['ld_rag']['ELASTIC_API']
}
category_terms_one = None
language_terms = None
category_terms_two = None
thresholds_dict = None
country_terms = None

########## APP start ###########
st.set_page_config(layout="wide")

# Get input parameters
st.markdown('### Please select search parameters 🔎')

# Get format and pull relevant prompt
format_choice = st.radio("Choose the preferred output format:", ['Summary', 'Alert'], index=None)
if format_choice == 'Alert':
    url = f'{os.environ["LANGSMITH_ACC"]}/simple-rag'
else:
    url = f'{os.environ["LANGSMITH_ACC"]}/simple-rag:9388b291'
    format_choice = 'Summary'
prompt_template = hub.pull(url)

selected_index = None
search_option = st.radio(
    "Choose Specific Indexes if you want to search one or more different indexes, choose All Project Indexes to select all indexes within a project.",
    ['Specific Indexes', 'All Project Indexes'])

if search_option == 'Specific Indexes':
    selected_indexes = st.multiselect('Please choose one or more indexes', flat_index_list, default=None,
                                      placeholder="Select one or more indexes")
    if selected_indexes:
        selected_index = ",".join(selected_indexes)
        st.write(f"We'll search in: {', '.join(selected_indexes)}")
    else:
        selected_index = None
else:
    project_choice = st.selectbox('Please choose a project', list(project_indexes.keys()), index=None,
                                  placeholder="Select project")
    if project_choice:
        selected_indexes = project_indexes[project_choice]
        selected_index = ",".join(selected_indexes)
        st.write(f"We'll search in: {', '.join(selected_indexes)}")

if selected_index:
    category_values_one, category_values_two, language_values, country_values = populate_default_values(selected_index,
                                                                                                        es_config)

    with st.popover("Tap to refine filters"):
        st.markdown("Hihi 👋")
        st.markdown(
            "If Any remains selected or no values at all, filtering will not be applied to this field. Start typing "
            "to find the option faster.")
        categories_one_selected = st.multiselect(
            'Select "Any" or choose one or more categories of the first (or only) level', category_values_one,
            default=['Any'])
        if "dem-arm" in selected_index:
            categories_two_selected = st.multiselect(
                'Select "Any" or choose one or more categories of the second level if those exist', category_values_two,
                default=['Any'])
        languages_selected = st.multiselect('Select "Any" or choose one or more languages', language_values,
                                            default=['Any'])
        countries_selected = st.multiselect('Select "Any" or choose one or more countries', country_values,
                                            default=['Any'])

    if "dem-arm" in selected_index:
        category_terms_one = populate_terms(categories_one_selected, 'misc.category_one.keyword')
        category_terms_two = populate_terms(categories_two_selected, 'misc.category_two.keyword')
    if "ru-balkans" in selected_index:
        category_terms_one = populate_terms(categories_one_selected, 'misc.category_one.keyword')
        category_terms_two = []
    else:
        category_terms_one = populate_terms(categories_one_selected, 'category.keyword')
        category_terms_two = []
    language_terms = populate_terms(language_values, 'language.keyword')
    country_terms = populate_terms(country_values, 'country.keyword')

    issues_fields = get_prefixed_fields(selected_index, 'issues.', es_config)

    with st.popover("Tap to define additional filtering by issue"):
        st.markdown("Edit at least one of the following thresholds to start filtering. "
                    "Any number of thresholds can be set simultaneously. "
                    "Editing several thresholds will result in filtering by at least one match to any of them (not all together).")
        if issues_fields:
            thresholds_dict = {}
            for field in issues_fields:
                min_value, max_value = st.slider(
                    f"Threshold range for {field}",
                    min_value=0.0,  # Set an appropriate minimum value
                    max_value=1.0,  # Set an appropriate maximum value
                    value=(0.0, 0.0),  # Default slider range
                    step=0.05  # Slider step increment
                )
                if (min_value, max_value) != (0.0, 0.0):
                    thresholds_dict[field] = f"{min_value}:{max_value}"

# Create prompt vector
input_question = None
st.markdown('### Please enter your question')
input_question = st.text_input("Enter your question here (phrased as if you ask a human)")

if input_question:

    formatted_start_date, formatted_end_date = None, None


    @st.cache_resource(hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
    def load_model():
        angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
                                            pooling_strategy='cls')
        return angle_model


    # Create question embedding
    angle = load_model()
    vec = angle.encode({'text': input_question}, to_numpy=True, prompt=Prompts.C)
    question_vector = vec.tolist()[0]

    # Get input dates
    selected_start_date = st.date_input("Select start date:")
    formatted_start_date = selected_start_date.strftime("%Y-%m-%d")
    st.write("You selected start date:", selected_start_date)
    selected_end_date = st.date_input("Select end date:")
    formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
    st.write("You selected end date:", selected_end_date)
    must_term = create_must_term(category_terms_one,
                                 category_terms_two,
                                 language_terms,
                                 country_terms,
                                 formatted_start_date=formatted_start_date,
                                 formatted_end_date=formatted_end_date,
                                 thresholds_dict=thresholds_dict)

    if formatted_start_date and formatted_end_date:

        # Authorise user
        if not check_password():
            st.stop()

        # Run search
        if st.button('RUN SEARCH'):
            start_time = time.time()
            max_doc_num = 30
            try:
                texts_list = []
                st.write(f'Running search for {max_doc_num} relevant posts for question: {input_question}')
                try:
                    es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                                       request_timeout=600)
                except Exception as e:
                    st.error(f'Failed to connect to Elasticsearch: {str(e)}')

                response = es.search(index=selected_index,
                                     size=max_doc_num,
                                     knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                          "query_vector": question_vector,
                                          "k": max_doc_num,
                                          "num_candidates": 10000,
                                          "filter": {
                                              "bool": {
                                                  "must": must_term,
                                                  "must_not": [{"term": {"type": "comment"}}]
                                              }
                                          }
                                          }
                                     )

                for doc in response['hits']['hits']:
                    texts_list.append((doc['_source']['translated_text'], doc['_source']['url']))

                st.write("Searching for documents, please wait...")

                # Format urls so they work properly within streamlit
                corrected_texts_list = [(text, 'https://' + url if not url.startswith('http://') and not url.startswith(
                    'https://') else url) for text, url in texts_list]

                # Get summary for the retrieved data
                customer_messages = prompt_template.format_messages(
                    question=input_question,
                    texts=corrected_texts_list)

                st.markdown(f'### This is {format_choice}, generated by GPT:')

                with callbacks.collect_runs() as cb:
                    st.write_stream(llm_chat.stream(customer_messages))
                    run_id = cb.traced_runs[0].id

                # st.markdown(content)
                st.write('******************')
                end_time = time.time()

                # Display tables
                st.markdown(f'### These are top {max_doc_num} texts used for alert generation:')
                df = create_dataframe_from_response(response)
                st.dataframe(df)
                display_distribution_charts(df, selected_index)

                # Send rating to Tally
                execution_time = round(end_time - start_time, 2)
                tally_form_url = f'https://tally.so/embed/wzq1Aa?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1&run_id={run_id}&time={execution_time}'
                components.iframe(tally_form_url, width=700, height=800, scrolling=True)

            except BadRequestError as e:
                st.error(f'Failed to execute search (embeddings might be missing for this index): {e.info}')
            except NotFoundError as e:
                st.error(f'Index not found: {e.info}')
            except Exception as e:
                st.error(f'An unknown error occurred: {str(e)}')
