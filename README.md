# Movie Recommendation System

This project is a movie recommendation system built using a dataset sourced from Kaggle, containing over one million movie records. The system employs various scripts for preprocessing, vector embedding, and model training to provide personalized movie recommendations.

## Dataset

The dataset used for this project is sourced from Kaggle and contains information on over one million movies. The dataset is utilized for training and building the recommendation system.

## Preprocessing

`pre-process.py`: This script is used to preprocess the dataset, combining all features into a single file and storing it in a new CSV file. This preprocessing step is essential for preparing the data for further analysis and model training.

## Vectorization and Database

`vector-db.py`: This script loads the preprocessed CSV file and performs text splitting, embedding, and vectorization using Faiss, a library for efficient similarity search and clustering of dense vectors.

## Model

`model.py`: This script contains the models used for movie recommendation. It includes:

- LLM Model (Llama2): A machine learning model designed and build by Meta.
- Sentence Transformer (MiniLM L6 V2): A transformer-based model trained on large-scale data for generating vector representations of sentences, including movie descriptions and user preferences.
- CTransformer: Python bindings for the Transformer models implemented in C/C++ using GGML library.

## Application

`app.py`: This script serves as the main entry point for executing the movie recommendation application. It utilizes Streamlit, a Python library for building interactive web applications, to create a user-friendly interface for accessing movie recommendations.

## Usage

1. Execute `pre-process.py` to preprocess the dataset and create the required CSV file. This step needs to be performed only once.
2. Execute the `vector-db.py` script to conduct vectorization and generate the vector database. This step should be performed only once, as it may take 4 to 6 hours depending on system specifications due to its time-consuming nature.
3. Launch the application by running `streamlit run app.py`. This will start the user interface, allowing users to receive personalized movie recommendations based on their preferences.

**Note:**

- Before executing app.py execute this command line (`pip install -r requirements.txt`)
- Ensure that `vector-db.py` is executed only once, as it can take a significant amount of time to complete.
- Use `streamlit run app.py` to run the application and interact with the recommendation system.

## Automated - EDA - Analysis

- Automated EDA Analysis with Python
- Automated EDA is used to analyze and understand the data in your dataset.
- It is a process of performing exploratory data analysis (EDA) on your dataset.
- EDA is used to identify patterns, outliers, and relationships in your data.
- EDA is used to identify trends, correlations, and relationships in your data.
- View the EDA Analysis results in HTML format.
- You will find that the EDA Analysis results in eda-analysis folder itself.

## Technologies used
- Python Programming -> Programming Language
- LangChain -> It is a framework to work with Large Language Model. Consist of Tensorflow and PyTorch libraries. Used for various operation such as chunk creation, embedding, building vector database, load and fine-tune LLM models and so on.
- Llama2 LLM -> Meta's LLM Model
- all-MiniLM-L6-v2 Sentence-Transformers -> Used for embedding and vector database
- CTransformer -> Loading and working with transformer. CTransformer and Transformer are two different things.
- HuggingFace Hub -> Use to get access for LLM models for text generation, text2text generation, sentence-similarity and so on.
- Faiss -> To Build Vector Database and store locally.
- Pinecone -> Similar to Faiss but it is cloud base vector database. limitations: free upto 100K vectors store. 
- Streamlit -> User interface and webview
- seaborn -> Data visualization
- sweetviz -> Automated EDA Analysis
- Pandas -> For pre-processing the data and cleaning of the data

## Dependencies

- Python 3.10
- Required libraries specified in `requirements.txt`
