# EV Charging Station Recommendation System

## Introduction
Our project aims to build a **search engine** for electric vehicle charging stations with personalized recommendations. The system consists of a naive ranker by distance, an ``DL-based l2r ranker``, a ``collaborative filtering`` recommender system, and a ``LLM-based personalization`` system. The performance of each ranker combination is evaluated with query results from Google Maps API processed with a customized relevance function. 

## Key features

- **Data scraping** using GoogleMap/Serp/NREL API
- **NLP** comments using ChatGPT API & **prompt engineering**
- LLM-based embedded vector space **personalization**.
- Item-item **reasoning** based on Collaborative Filtering & Relevance Matrix
- **“Learn to Rank”** using LightGBM, automatic pesudo-label generation using ChatGPT & GoogleMap API.
- Integrated mainstream rankers such as **BM25**(core feature of Bing), TF-IDF, Pivoted Normalization, DirichletLM.
- **MEM-efficient** data structure based on inverted index
- **User Interface** based on Flask, GoogleMAP API

## Methodology

### Data Collection





## Reference

[1] Y. Kobayashi, N. Kiyama, H. Aoshima and M. Kashiyama, "A route search method for electric vehicles in consideration of range and locations of charging stations," 2011 IEEE Intelligent Vehicles Symposium (IV), Baden-Baden, Germany, 2011, pp. 920-925, doi: 10.1109/IVS.2011.5940556.

[2] Guillet Marianne, Hiermann Gerhard, Kroller Alexander, Schiffer Maximilian. Electric Vehicle Charging Station Search in Stochastic Environments. Transportation Science, vol 56, pp 483-500, 2022.  doi: 10.1287/trsc.2021.1102

[3] Information Retrieval in the Commentsphere, ACM Transactions on Intelligent Systems and Technology, Volume 3, Issue 4, Article No.: 68pp 1–21
https://doi.org/10.1145/2337542.2337553

[4] Shen, Chenhui, Cheng, Liying, Nguyen, Xuan-Phi, You Yang, and Bing, Lidong. Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization. 2023. https://arxiv.org/abs/2305.13091

[5] Evangelia Christakopoulou and George Karypis. 2016. Local Item-Item Models For Top-N Recommendation. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys '16). Association for Computing Machinery, New York, NY, USA, 67–74. https://doi.org/10.1145/2959100.2959185
