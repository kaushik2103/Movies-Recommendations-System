# This file was used to pre-process the data, the file was downloaded from the Kaggle dataset
# No need to run this file every time. It is use only once to update the data.
# Kaggle dataset: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?resource=download
# raw_file: data/movie_dataset.csv -> from Kaggle
# preprocessed_file: data/pre-processed_dataset.csv -> for the further uses

# Import necessary libraries
import pandas as pd


# Define the class for preprocessing the data
class DataPreprocessor:
    # Define the constructor
    def __init__(self, raw_file, preprocessed_file):
        self.raw_file = raw_file
        self.preprocessed_file = preprocessed_file

    # Define the method for preprocessing the data
    def preprocess_data(self):
        # Load the data
        print("Loading data...")
        movie_dataset = pd.read_csv(self.raw_file)
        # Combine the information
        print("Combining information...")
        movie_dataset["combined_info"] = movie_dataset.apply(lambda row: self._combine_info(row), axis=1)
        # Save the data
        print("Saving data...")
        movie_dataset["combined_info"].to_excel(self.preprocessed_file, index=False, header=True)

    # Define the method for combining the information
    @staticmethod
    def _combine_info(row):
        return f"Title: {row['title']}. Overview: {row['overview']} Genres: {row['genres']} Vote: {row['vote_average']} Release: {row['release_date']} Revenue: {row['revenue']} Adult: {row['adult']} Budget: {row['budget']} Original title: {row['original_title']} Popularity: {row['popularity']} Production: {row['production_companies']} Languages: {row['spoken_languages']}"


# Create an instance of the DataPreprocessor class and call the preprocess_data method
if __name__ == "__main__":
    preprocessor = DataPreprocessor(raw_file="data/movie_dataset.csv",
                                    preprocessed_file="data/pre-processed_dataset.xlsx")
    preprocessor.preprocess_data()
