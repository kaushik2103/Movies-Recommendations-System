# This file contains the code for the movie recommendation system.
# It contains LLM model and retrieval chain
# We have used the Llama2 7B model for Large Language Models
# We have used the ALL-MiniLM-L6-v2 model for embedding the data in the vector database
# Instead of Transformer model, we used CTransformers model.

# Import necessary libraries
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA


# Define the class for movie recommendation system
class MovieRecommendationSystem:
    # Define the constructor
    def __init__(self):
        self.DB_FAISS_PATH = 'vectorstore/db_faiss'
        self.custom_prompt_template = """You are a movie recommender system that help users to find movies that match their preferences. 
        Use the following pieces of context to answer the question at the end. 
        For each question, suggest more movies, with a short description of the plot and the reason why the user might like it.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Your response:"""
        self.qa = self.qa_bot()

    # Define the methods for setting the custom prompt
    def set_custom_prompt(self):
        prompt = PromptTemplate(template=self.custom_prompt_template, input_variables=['context', 'question'])
        return prompt

    # Define the methods for retrieval
    @staticmethod
    def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': prompt})
        return qa_chain

    # Define the methods for loading the LLM
    @staticmethod
    def load_llm():
        llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=500,
                            temperature=0.8)
        return llm

    # Define the methods for QA
    # Load the vector database
    # Load the LLM
    # Load the QA
    # Load embeddings
    def qa_bot(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(self.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = self.load_llm()
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain(llm, qa_prompt, db)
        return qa

    # Define the methods for final result
    def final_result(self, query):
        response = self.qa({"query": query})
        return response.get("result")

    # Define the methods for running
    # Set the page configuration for the streamlit
    # Set the title of the page
    # Set the text input
    # Set the response
    # Set the error message
    def run(self):
        st.set_page_config(page_title="Movie Recommendation System", page_icon=":robot:")
        st.title("Movie Recommendation System")
        query = st.text_input("Please enter what type of movie you like: ")
        if query:
            answer = self.final_result(query)
            st.write("Response")
            st.write(answer)
        else:
            st.write("Please enter what type of movie you want!")


# Define the main function
if __name__ == "__main__":
    bot = MovieRecommendationSystem()
    bot.run()
