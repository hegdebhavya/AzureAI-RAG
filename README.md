# RAG using AzureopenAI

Azure OpenAI and LangChain to create a vector database and perform question-answering tasks in terminal and using streamlit. 

## Prerequisites

Aceess to a Azure subscription to deploy azure openAI endpoint and to deploy embedding model

## Steps

* Create and activate virtual environment </br>

``` pip install virtualenv ``` </br>
``` python -m venv myenv ``` </br>
``` myenv\Scripts\activate ``` </br>

* Install the necessary pip packages from the requirements file </br>

```pip install -r .\requirements.txt ```


* Run vector dB script to create the embeddings for the text file (will be visible in the faiss-db folder if successful) </br>
``` py vector_db_creator.py ```

* Run the main script to interact with the chatbot via terminal </br>

``` py main.py ``` </br>

* To interact with the chatbot via UI, run the streamlit script using the streamlit command



``` streamlit run streamlit.py ``` </br>

 * Deactivate virtual environment 

``` deactivate ```
