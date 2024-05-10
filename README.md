# EV Charging Station Recommendation System

## Introduction
Our project aims to build a **search engine** for electric vehicle charging stations with personalized recommendations. The system consists of a naive ranker by distance, and a ``LLM-based personalization`` system. The performance of each ranker combination is evaluated with query results from Google Maps API processed with a customized relevance function. 

## Start the Server
First, initialize the database by running the following command in the terminal:  
```flask --app electrifind init-db```

To start the server, run the following command in the terminal:  
```flask --app electrifind run --debug```   

To start the server with auto reload, run the following command in the terminal:  
```waitress-serve --call 'electrifind:create_app'```

The query contains the following fields:
- `latitude`: The latitude of the user.
- `longitude`: The longitude of the user.
- `user_id`: The user ID.

The system will return the top electric vehicle charging stations based on the user's preferences.

## Environment
To use the syste, you need to first create a Python virtual environment:  
```python3 -m venv .venv```  

Switch to the virtual environment:  
```source .venv/bin/activate```  

Install the required packages:  
```pip install -r requirements.txt```

## Key Features
- **Data scraping** using GoogleMap/Serp/NREL API
- **NLP** comments using ChatGPT API & **prompt engineering**
- LLM-based embedded vector space **personalization**.

## Project Layout
```
/project2-team7
├── .venv/
├── data/
│   ├── encoded_station.npy
│   ├── encoded_user_profile.npy
│   ├── GPT_analysis_AA_DTW.csv
│   ├── mergd_reviews_and_predicted_dataset.csv
│   ├── NREL_RAW.csv
│   ├── row_to_docid.txt
│   ├── station_personalized_features.csv
├── src/
│   ├── __init__.py
│   ├── models.py: Models for the system
│   ├── pipeline.py: Pipeline for the whole system, use it as main!
│   ├── ranker.py: Rank the documents
│   ├── utils.py: Utility functions
│   ├── vector_ranker.py: Vector ranker
├── notebook/
|   ├── Deep Learning Factorization model.ipynb
│   ├── evaluation.ipynb
|   ├── Google_Map_Find_API.ipynb
│   ├── Google_Map_Review_API.ipynb
│   ├── GPT_prompt.ipynb
│   ├── GPT_sentiment_analysis.ipynb
│   ├── Joint_Data_Model.ipynb
│   ├── NREL_processing_numerical.ipynb
│   ├── NREL_processing.ipynb
│   ├── Review_Feature_Extractor.ipynb
│   ├── user_profile.ipynb
├── .gitignore
├── README.md
├── requirements.txt
```

## Methodology

### Data Collection

- `encoded_station.npy`:
This file contains the encoded data of the electric vehicle charging stations in the United States.

- `encoded_user_profile.npy`:
This file contains the encoded data of the user profiles.  

- `GPT_analysis_AA_DTW.csv`:
This file contains the metadata and score of the electric vehicle charging stations in the United States.

- `mergd_reviews_and_predicted_dataset.csv`:
This file contains the merged data after LLM and GPT analysis.

- `NREL_raw.csv`:
This file contains raw data from NREL API of electric vehicle charging stations in the United States.

- `row_to_docid.txt`:
This file contains the mapping of the row index to the document ID.

- `station_personalized_features.csv`:
This file contains the personalized features of the electric vehicle charging stations in the United States.

### Notebooks
- `Google_Map_Review_API.ipynb`:
This notebook contains the code for scraping the reviews of the electric vehicle charging stations in the United States using the Google Map API.

- `GPT_prompt.ipynb`:
This notebook contains the code for analyzing the reviews of the electric vehicle charging stations in the United States using the ChatGPT API.