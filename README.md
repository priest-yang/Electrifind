# EV Charging Station Recommendation System

## Introduction
Our project aims to build a **search engine** for electric vehicle charging stations with personalized recommendations. The system consists of a naive ranker by distance, an ``DL-based l2r ranker``, a ``collaborative filtering`` recommender system, and a ``LLM-based personalization`` system. The performance of each ranker combination is evaluated with query results from Google Maps API processed with a customized relevance function. 

## Start the Server
First, initialize the database by running the following command in the terminal:  
```flask --app electrifind init-db```

To start the server, run the following command in the terminal:  
```waitress-serve --call 'electricind:create_app'```

## Key Features

- **Data scraping** using GoogleMap/Serp/NREL API
- **NLP** comments using ChatGPT API & **prompt engineering**
- LLM-based embedded vector space **personalization**.
- Item-item **reasoning** based on Collaborative Filtering & Relevance Matrix
- **“Learn to Rank”** using LightGBM, automatic pesudo-label generation using ChatGPT & GoogleMap API.
- Integrated mainstream rankers such as **BM25**(core feature of Bing), TF-IDF, Pivoted Normalization, DirichletLM.
- **MEM-efficient** data structure based on inverted index
- **User Interface** based on Flask, GoogleMAP API

## Project Layout
```
/electrifind
├── .venv/
├── archive/
├── data/
│   ├── docid_NREL_map.csv: Map between docid and NREL id
│   ├── NREL_processed.csv
│   ├── NREL_raw.csv
│   ├── relevance.train.csv
│   ├── relevance.test.csv
├── electrifind/
├── notebook/
│   ├── NREL_processing.ipynb
│   ├── Google_Map_Find_API.ipynb
├── src/
│   ├── document_preprocessor.py: Preprocess the documents
│   ├── indexing.py: Indexing the documents
│   ├── ranker.py: Rank the documents
│   ├── relevance.py: Relevance testing
│   ├── l2r.py: Learning to rank
│   ├── utils.py: Utility functions
│   ├── cf.py: Collaborative filtering
│   ├── vector_ranker.py: Vector ranker
│   ├── pipeline.py: Pipeline for the whole system
├── tests/
├── .gitignore
├── pyproject.toml
├── README.md
```

## Methodology

### Data Collection

- `NREL_raw.csv`:
This file contains raw data from NREL API.  

- `NREL_corpus.jsonl`:
This file contains descriptions of charging stations from NREL API. 

- `NREL_processed.csv`:
This file contains preprocessed data from NREL API, which is used as the data source for the search engine.  

- `relevance.train.csv`:
This file contains the relevance data for the training of the l2r ranker.

- `relevance.test.csv`:
This file contains the relevance data for the testing of the l2r ranker.

### Notebooks

- `NREL_processing.ipynb`:
This notebook contains the code for data processing from NREL API into the corpus for the search engine. 

- `NREL_processing_numerical.ipynb`:
This notebook contains the code for data processing from NREL API into the numerical dataframe for the search engine.

- `Google_Map_Find_API.ipynb`:
This notebook creates the relevance data for the training and testing of the l2r ranker.

- `evaluation.ipynb`:  
This notebook contains the code for calling a search engine object and evaluating the perforance by setting all the parameters. 

## Reference

[1] Y. Kobayashi, N. Kiyama, H. Aoshima and M. Kashiyama, "A route search method for electric vehicles in consideration of range and locations of charging stations," 2011 IEEE Intelligent Vehicles Symposium (IV), Baden-Baden, Germany, 2011, pp. 920-925, doi: 10.1109/IVS.2011.5940556.

[2] Guillet Marianne, Hiermann Gerhard, Kroller Alexander, Schiffer Maximilian. Electric Vehicle Charging Station Search in Stochastic Environments. Transportation Science, vol 56, pp 483-500, 2022.  doi: 10.1287/trsc.2021.1102

[3] Information Retrieval in the Commentsphere, ACM Transactions on Intelligent Systems and Technology, Volume 3, Issue 4, Article No.: 68pp 1–21
https://doi.org/10.1145/2337542.2337553

[4] Shen, Chenhui, Cheng, Liying, Nguyen, Xuan-Phi, You Yang, and Bing, Lidong. Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization. 2023. https://arxiv.org/abs/2305.13091

[5] Evangelia Christakopoulou and George Karypis. 2016. Local Item-Item Models For Top-N Recommendation. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys '16). Association for Computing Machinery, New York, NY, USA, 67–74. https://doi.org/10.1145/2959100.2959185
