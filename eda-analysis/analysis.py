# Automated EDA Analysis with Python
# Automated EDA is used to analyze and understand the data in your dataset.
# It is a process of performing exploratory data analysis (EDA) on your dataset.
# EDA is used to identify patterns, outliers, and relationships in your data.
# EDA is used to identify trends, correlations, and relationships in your data.
# View the EDA Analysis results in HTML format.
# You will find that the EDA Analysis results in eda-analysis folder itself.

# Import necessary libraries
import seaborn as sns
import pandas as pd

# Using SweetViz Library for EDA Analysis
import sweetviz as sv

movie_dataset = pd.read_csv("../data/movie_dataset.csv")
preprocessed_dataset = pd.read_csv("../data/pre-processed_dataset.csv")

movie_dataset = sv.analyze(movie_dataset)
preprocessed_dataset = sv.analyze(preprocessed_dataset)
movie_dataset.show_html("movie_dataset.html")
preprocessed_dataset.show_html("preprocessed_dataset.html")

