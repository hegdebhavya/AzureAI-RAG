# RAG using Azure OpenAI

This project utilizes Azure OpenAI and LangChain to create a vector database and perform question-answering tasks in the terminal and using Streamlit.

## Prerequisites

- Access to an Azure subscription to deploy the Azure OpenAI endpoint and the embedding model.

## Steps

1. **Create and activate a virtual environment:**

    ```sh
    pip install virtualenv
    python -m venv myenv
    myenv\Scripts\activate
    ```

2. **Install the necessary pip packages from the requirements file:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Run the vector database script to create embeddings for the text file:** 

   (The embeddings will be stored in the `faiss-db` folder if successful)

    ```sh
    python vector_db_creator.py
    ```

4. **Run the main script to interact with the chatbot via the terminal:**

    ```sh
    python main.py
    ```

5. **To interact with the chatbot via the UI, run the Streamlit script using the Streamlit command:**

    ```sh
    streamlit run streamlit.py
    ```

6. **Deactivate the virtual environment:**

    ```sh
    deactivate
    ```

## Usage

This project can be used to perform question-answering tasks using a custom-trained vector database. The terminal interaction provides a command-line interface, while the Streamlit script offers a graphical user interface.


## Contact

If you have any questions or feedback, feel free to contact me at hegdeb09@gmail.com
