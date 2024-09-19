# This file contains the code for executing the movie recommendation system.
# This is the main file for executing the movie recommendation system.
# To execute the movie recommendation system, run the following command:
# streamlit run app.py

from model import MovieRecommendationSystem

if __name__ == "__main__":
    bot = MovieRecommendationSystem()
    bot.run()
