# **Technology**

With retrieval-augmented generation, users can have direct conversations with data repositories. When the user formulates the question (query), retrieval components match it to billions of records available in our Elastic indexes based on the semantic similarity, and if relevant records are found, return up to a predefined (but configurable) number of such relevant records.

Then these relevant records are provided to the LLM (Large Language Model), which is configured to ultimately rely only on the provided data. At the generation stage of retrieval-augmented generation, LLM formulates the answer to the user’s question in a human-readable format and provides citations. 

To further increase relevance users can narrow down the search parameters, so that the model is looking for relevant documents only on some specific indexes, and timeframe, with user-defined author category, language etc.

Source data (the retrieved records themselves) are provided for the user’s convenience right below the generated answer, this table is available for sorting and download. Some default aggregations are also provided - visualisations for country, language and category distribution.

- Retrieval Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources. [Read more](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/).
- Semantic Similarity is a metric, calculated by similarity models which convert input texts into vectors (embeddings) that capture semantic information and calculate how close (similar) they are. [Read more.](https://huggingface.co/tasks/sentence-similarity)
- LLM is a type of artificial intelligence (AI) program that can recognise and generate text, among other tasks. LLMs are trained on huge sets of data — hence the name "large." LLMs are built on machine learning: specifically, a type of neural network called a transformer model. [Read more.](https://www.cloudflare.com/en-gb/learning/ai/what-is-large-language-model/)

# Data

The underlying data, from which relevant documents are selected is stored in our distributed search and analytics engine - Elastic. The data is pre-loaded, so user’s questions are not directed towards social platforms or websites themselves, but the copy of their relevant records, that we store in Elastic. After loading, data is enriched to be usable by our multiple analytical instruments.

Normally, within the project data is stored in per-platform indexes (sub-storages), so for Dem By, for example, we can maintain indexes dem-by-telegram, dem-by-vkontakte, dem-by-telegram etc. Our app lets users select particular indexes (platforms), or all indexes within the particular project.

## Default data filters

**Platform**

Select all platforms (indexes) within the project, or choose any number of platforms of interest among available options with multi-select.

**Date**

Select the time interval in the provided date input fields. Some default values are used for users’ convenience, but time intervals can be expanded or narrowed down as needed.

**Categories (tap to select)**

Select one or several particular author categories (assigned at the data enrichment stage after information space mapping). If “Any” is present among the selected options, the system will always return all categories: to narrow down the search please remove any from the selection.

**Countries (tap to select)**

Select one or several countries of author origin or main operation. If “Any” is present among the selected options, the system will always return all countries: to narrow down the search please remove any from the selection.

**Languages (tap to select)**

Select one or several countries of author origin or main operation. If “Any” is present among the selected options, the system will always return all languages: to narrow down the search please remove any from the selection.

# FAQ & Best Practices

While using the app is designed to be intuitive, we provide several suggestions for the best results.

### How to formulate questions and apply filtering

- Please, only use English language for your question.
- Shorter questions (up to 10 words) work best.
- Narrowing down the search with filters can benefit you if your question is of significant scope (you expect a lot of relevant data). Targeting certain author categories or languages can be beneficial in this setting. If you expect your question to be of limited scope, start with less to no filtering and based on retrieved results narrow down as needed.
- Selecting a larger timeframe means that the model might need more time to select relevant documents. Normal response time varies between 15 and 60 seconds depending on the timeframe, how much relevant data the model is selecting from and the length of the retrieved documents.

### How to interpret and validate results

- We always provide retrieved relevant records as a table, available for investigation and download.
- While it’s not feasible to retrieve **all** the relevant documents (the count of relevant documents can be as low as 0 and as high as tens of thousands of documents), we make sure to retrieve the **most** relevant ones. That means that some documents of lower relevance can be left out and that is the expected behaviour of the system.
- The summary answer provided by the model is required to contain citations for improved accountability. These are the same documents as the ones provided in the table.
- While we provide source and text to the LLM to generate the answer, it does not yet have access to the rest of the parameters (like actor category). So asking the model to compare stances for different countries or author categories would not provide reliable results (these parameters can be made available for the summary upon further discussion).

# **Troubleshooting and Support**

### **Common issues**

1️⃣ No results are provided (the table is empty and the answer generated by the model indicated it did not get any input). 

Solutions: 

- Widen your search parameters. It is possible that the selected search space does not have relevant records.
- Try rephrasing the question.
- If you suspect that the search should not return empty results, please report this issue.

2️⃣ The app is sleeping (you see ”This app has gone to sleep due to inactivity. Would you like to wake it back up?”)

Solution: Press the “Yes, get this app back up!”. This is expected behaviour - when not in use for an extended period, the app goes to sleep to efficiently redistribute resource usage.

3️⃣ You see an error in red.

Solution: Try reloading the page. If that does not solve the issue, please [submit your request by filling the Request Form out (type of request - LD app Technical issue)](https://docs.google.com/forms/d/e/1FAIpQLSfZTr4YoXXsjOOIAMVGYCeGgXd6LOsCQusctJ7hZODaW5HzGQ/viewform?usp=sf_link).

### Feedback and support

1️⃣ To report any technical issues, please [submit your request by filling the Request Form out (type of request - LD app Technical issue)](https://docs.google.com/forms/d/e/1FAIpQLSfZTr4YoXXsjOOIAMVGYCeGgXd6LOsCQusctJ7hZODaW5HzGQ/viewform?usp=sf_link).

2️⃣ At the bottom of the page (after the search results) the Feedback Form for your evaluation of the app’s performance is available. We value your feedback.