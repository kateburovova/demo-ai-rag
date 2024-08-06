# Base
import time
import logging
import random

from datetime import datetime, timedelta

# Internal
from authentificate import check_password
from utils import (display_distribution_charts, populate_default_values,
                   populate_terms, create_must_term, create_dataframe_from_response,
                   get_prefixed_fields, set_state_defaults, load_config, load_es_config,
                   get_texts_from_elastic, get_guestion_vector, init_llms, get_keys, generate_output_stream,
                   init_langsmith_params, pull_prompts, get_default_date_range, get_topic_counts,
                   infer_topic_index_names, get_summary_and_narratives)

# External
import streamlit as st
import streamlit.components.v1 as components

# Setting basic configurations

logging.basicConfig(level=logging.INFO)
es_config = load_es_config()
config = load_config()
api_keys = get_keys()
llm_models = init_llms(config, api_keys)
init_langsmith_params(config)
prompts = pull_prompts(config)
set_state_defaults()

########## APP start ###########
st.set_page_config(layout="wide")

logo_url = 'assets/Blue_black_long.png'
st.image(logo_url, width=300)

if not check_password():
    st.stop()

# description
st.markdown('App relies on data, collected and enriched by our team and provides citations for all sources used for '
            'answers. \n'
            'If you are running this app from a mobile device, tap on any '
            'empty space to apply changes to input fields. '
            'If you experience any technical issues, please [submit the form](https://docs.google.com/forms/d/e/1FAIpQLSfZTr4YoXXsjOOIAMVGYCeGgXd6LOsCQusctJ7hZODaW5HzGQ/viewform?pli=1) by selecting "LD app Technical Issue" for '
            'the type of request. To give feedback for request output, '
            'please use the feedback form at the end of the page.')

with open('assets/How_to.md', 'r') as file:
    markdown_content = file.read()

with st.expander("Learn more about the app"):
    st.markdown(markdown_content, unsafe_allow_html=True)

# Get input parameters
st.markdown('### Please select search parameters 🔎')

# Get format and pull relevant prompt
task_options = list(config['tasks'].keys())
label_options = [config['tasks'][task]['label'] for task in task_options]
selected_label = st.radio("Choose the preferred output format:", label_options)
selected_task = task_options[label_options.index(selected_label)]

if selected_task == 'actor_comparison':
    st.session_state.compare_categories = True

# Offer comparison mode if enabled in config
if config['comparison_mode']['enabled']:
    comparison_mode = st.checkbox("Enable comparison mode")
else:
    comparison_mode = False

search_option = st.radio(
    "Choose Specific Indexes if you want to search one or more different indexes, choose All Project Indexes to select all indexes within a project.",
    ['All Project Indexes', 'Specific Indexes'])

if search_option == 'Specific Indexes':
    flat_index_list = [index for project in config['project_indexes'].values() for index in project]
    selected_indexes = st.multiselect('Please choose one or more indexes', flat_index_list, default=None,
                                      placeholder="Select one or more indexes")
    if selected_indexes:
        st.session_state.selected_index = ",".join(selected_indexes)
        st.write(f"We'll search in: {', '.join(selected_indexes)}")
    else:
        st.session_state.selected_index = None
else:
    if len(config['project_indexes']) == 1:  # If we have only 1 project, we don't offer choice of projects
        selected_indexes = list(config['project_indexes'].values())[0]
        st.session_state.selected_index = ",".join(selected_indexes)
        st.write(f"We'll search in: {', '.join(selected_indexes)}")

    else:
        project_choice = st.selectbox('Please choose a project', list(config['project_indexes'].keys()), index=None,
                                      placeholder="Select project")
        if project_choice:
            selected_indexes = config['project_indexes'][project_choice]
            st.session_state.selected_index = ",".join(selected_indexes)
            st.write(f"We'll search in: {', '.join(selected_indexes)}")

if st.session_state.selected_index:
    category_values_one, category_values_two, language_values, country_values \
        = populate_default_values(st.session_state.selected_index, es_config)

    issues_fields = get_prefixed_fields(st.session_state.selected_index, 'issues.', es_config)

    with st.form("Tap to refine filters"):
        st.markdown("Hihi 👋")
        st.markdown(
            "If Any remains selected or no values at all, filtering will not be applied to this field. Start typing "
            "to find the option faster.")

        min_date, default_start_date, default_end_date, today = get_default_date_range(config)
        default_non_null_categories_one = [category for category in category_values_one
                                           if category not in ['Any', None, '']]
        logging.info(f"Default categories_one: {default_non_null_categories_one}")

        if "dem-arm" or "ru-balkans" in st.session_state.selected_index:
            st.session_state.use_base_category = False
        else:
            st.session_state.use_base_category = True

        if st.session_state.compare_categories and st.session_state.use_base_category:
            st.session_state.category_terms_one = populate_terms(default_non_null_categories_one,
                                                                 'category.keyword')
        elif st.session_state.compare_categories and not st.session_state.use_base_category:
            st.session_state.category_terms_one = populate_terms(default_non_null_categories_one,
                                                                 'misc.category_one.keyword')

        date_range = st.date_input(
            "Select date range",
            value=(default_start_date, default_end_date),
            min_value=min_date,
            max_value=datetime.now() + timedelta(days=365),
        )

        if len(date_range) == 2:
            selected_start_date, selected_end_date = date_range
            st.session_state.formatted_start_date = selected_start_date.strftime("%Y-%m-%d")
            st.session_state.formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
        else:
            st.session_state.formatted_start_date = None
            st.session_state.formatted_end_date = None

        if st.session_state.compare_categories:
            categories_one_selected = st.multiselect(
                'Select "Any" or choose one or more categories of the first (or only) level', category_values_one,
                default=default_non_null_categories_one)
        else:
            categories_one_selected = st.multiselect(
                'Select "Any" or choose one or more categories of the first (or only) level', category_values_one,
                default=['Any'])
        if "dem-arm" in st.session_state.selected_index:
            categories_two_selected = st.multiselect(
                'Select "Any" or choose one or more categories of the second level if those exist', category_values_two,
                default=['Any'])
        languages_selected = st.multiselect('Select "Any" or choose one or more languages', language_values,
                                            default=['Any'])
        logging.info(f"Selected languages: {language_values}")
        countries_selected = st.multiselect('Select "Any" or choose one or more countries', country_values,
                                            default=['Any'])
        logging.info(f"Selected countries: {country_values}")

        submitted = st.form_submit_button("Save my choice", type="primary")

        if submitted:
            if "dem-arm" in st.session_state.selected_index:
                st.session_state.category_terms_one = populate_terms(categories_one_selected,
                                                                     'misc.category_one.keyword')
                st.session_state.category_terms_two = populate_terms(categories_two_selected,
                                                                     'misc.category_two.keyword')
            elif "ru-balkans" in st.session_state.selected_index:
                st.session_state.category_terms_one = populate_terms(categories_one_selected,
                                                                     'misc.category_one.keyword')
                st.session_state.category_terms_two = []
            else:
                st.session_state.category_terms_one = populate_terms(categories_one_selected, 'category.keyword')
                logging.info(f"Base category use: {st.session_state.use_base_category}")
                st.session_state.category_terms_two = []

            st.session_state.language_terms = populate_terms(languages_selected, 'language_text.keyword')
            st.session_state.country_terms = populate_terms(countries_selected, 'country.keyword')
            logging.info(f"1. Session state: {st.session_state}")

            logging.info(
                f"Category terms after population: one={st.session_state.category_terms_one}, "
                f"two={st.session_state.category_terms_two}")
            logging.info(f"Language terms: {st.session_state.language_terms}")
            logging.info(f"Country terms: {st.session_state.country_terms}")
else:
    issues_fields = None

logging.info(f"2. Session state: {st.session_state}")

if issues_fields:
    if st.button('Click to define issues') or st.session_state.show_issues_form:
        st.session_state.show_issues_form = True
        with st.form("Tap to define additional filtering by issue"):
            st.markdown("Edit at least one of the following thresholds to start filtering. "
                        "Any number of thresholds can be set simultaneously. "
                        "Editing several thresholds will result in filtering by at least one match to any of them (not all together).")

            for field in issues_fields:
                # Use session state to remember the slider values
                if field not in st.session_state:
                    st.session_state[field] = (0.0, 0.0)

                min_value, max_value = st.slider(
                    f"Threshold range for {field}",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state[field],
                    step=0.05
                )
                # Update session state
                st.session_state[field] = (min_value, max_value)

            submitted_issues = st.form_submit_button("Save my choice", type="primary")
            if submitted_issues:
                st.session_state.thresholds_dict = {
                    field: f"{st.session_state[field][0]}:{st.session_state[field][1]}"
                    for field in issues_fields
                    if st.session_state[field] != (0.0, 0.0)
                }
                logging.info(f"Issue terms: {st.session_state.thresholds_dict}")
    logging.info(f"3. Session state: {st.session_state}")

# Create prompt vector
st.markdown('### Please enter your question')
input_question = st.text_input("Enter your question here (phrased as if you ask a human)")
logging.info(f"4. Session state: {st.session_state}")

if input_question:
    question_vector = get_guestion_vector(input_question)

    logging.info(
        f"Selected categories: one={st.session_state.category_terms_one}, two={st.session_state.category_terms_two}")
    logging.info(f"Selected languages: {st.session_state.language_terms}")
    logging.info(f"Selected countries: {st.session_state.country_terms}")
    logging.info(f"Issue terms after question definition: {st.session_state.thresholds_dict}")

    if st.session_state.formatted_start_date and st.session_state.formatted_end_date:
        must_term = create_must_term(st.session_state.category_terms_one,
                                     st.session_state.category_terms_two,
                                     st.session_state.language_terms,
                                     st.session_state.country_terms,
                                     formatted_start_date=st.session_state.formatted_start_date,
                                     formatted_end_date=st.session_state.formatted_end_date,
                                     thresholds_dict=st.session_state.thresholds_dict)
    if must_term:

        # Run search
        if st.button('RUN SEARCH', type="primary"):
            corrected_texts_list, response = get_texts_from_elastic(input_question=input_question,
                                                                    question_vector=question_vector,
                                                                    must_term=must_term,
                                                                    es_config=es_config,
                                                                    config=config)
            prompt_template = prompts[selected_task]

            st.markdown(f'### This is {selected_label}, generated based on relevant data:')

            if comparison_mode:

                start_time = time.time()
                # Randomly select two models for comparison
                model_names = list(llm_models.keys())
                if len(model_names) < 2:
                    st.error("Not enough models available for comparison.")
                else:
                    model1, model2 = random.sample(model_names, 2)

                    # st.write(f"Comparing models: {model1} and {model2}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"## {model1}")
                        placeholder1 = st.empty()
                        content1, run_id1 = generate_output_stream(prompt_template, llm_models[model1],
                                                                   corrected_texts_list, placeholder1, input_question)

                    with col2:
                        st.markdown(f"## {model2}")
                        placeholder2 = st.empty()
                        content2, run_id2 = generate_output_stream(prompt_template, llm_models[model2],
                                                                   corrected_texts_list, placeholder2, input_question)

                end_time = time.time()
                voting_form_url = f'https://tally.so/embed/{config["tally_form"]["voting_id"]}?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1&model1_id={model1}&model2_id={model2}'
                components.iframe(voting_form_url,
                                  width=config['tally_form']['voting_width'],
                                  height=config['tally_form']['voting_height'],
                                  scrolling=True)


            else:
                start_time = time.time()
                placeholder = st.empty()
                content, run_id = generate_output_stream(prompt_template,
                                                         llm_models[config['llm']['default_model']],
                                                         corrected_texts_list,
                                                         placeholder, input_question)
                end_time = time.time()

            # Display tables
            st.markdown(f'### These are top {config["max_doc_num"]} texts used for generation:')
            df = create_dataframe_from_response(response)
            st.dataframe(df)
            display_distribution_charts(df, st.session_state.selected_index)

            # Display summary and narratives for featured topics
            topic_count_df = get_topic_counts(response)
            if not topic_count_df.empty:
                topic_indexes = infer_topic_index_names(st.session_state.selected_index)
                summary_topic_df = get_summary_and_narratives(topic_count_df, topic_indexes, es_config)
                st.markdown("### Topics")
                st.dataframe(summary_topic_df)
            else:
                st.write('#### No topics with summaries were found.')

            if config['debug']['display_source_texts']:
                st.markdown("### Raw Data for Copying:")
                raw_text = str(corrected_texts_list)
                st.text_area("Copy this data:", value=raw_text, height=300)

            # Send rating to Tally
            if not comparison_mode:
                execution_time = round(end_time - start_time, 2)
                tally_form_url = f'https://tally.so/embed/{config["tally_form"]["feedback_id"]}?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1&run_id={run_id}&time={execution_time}'
                components.iframe(tally_form_url,
                                  width=config['tally_form']['feedback_width'],
                                  height=config['tally_form']['feedback_height'],
                                  scrolling=True)

    if st.button('RE-RUN APP'):
        time.sleep(1)
