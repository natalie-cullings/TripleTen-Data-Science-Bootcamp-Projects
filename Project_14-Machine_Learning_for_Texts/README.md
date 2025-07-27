# Project 14: Machine Learning for Texts
#### *Movie Review Sentiment Classification Model*

## Table of Contents

1. [Overview & Objective](#overview_objective)
2. [Dataset](#dataset)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Overview & Objective <a id="overview_objective"></a>

The Film Junky Union, a (hypothetical) new, edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews.

The objective of this project is to train a model to automatically detect negative reviews. Using a dataset of IMBD movie reviews with polarity labeling, the model will classify positive and negative reviews. It needs to have an F1 score of at least 0.85.


## Dataset <a id="dataset"></a>

This project relies on the provided `imdb_reviews.tsv` data file, which contains the following fields:

**Features**
 - **tconst:** The unique ID of the title.
 - **title_type:** The type of the title (short, video, movie, TV episode, etc.).
 - **primary_title:** The primary title.
 - **original_title:**	The original title.
 - **start_year:** The start year of the title.
 - **end_year:** The end year of the title.
 - **runtime_minutes:**	The length of the title in minutes.
 - **is_adult:** Whether the title is adult in content or not.
 - **genres:**	The genres of the title.
 - **average_rating:**	The average rating of the title.
 - **votes:** The number of votes the title received.
 - **review:**	The text of the review.
 - **rating:**	The rating of the title.
 - **sp:**	The sentiment polarity label for whether the review is positive or negative.
 - **ds_part:**	Whether the record is part of the 'train' or 'test' part of the dataset.
 - **idx:** The index of the review in the dataset.
**Target**
 - **pos:**	The sentiment for the review where '0' represents a negative review and '1' represents a positive review.

The data was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and EDA:**
   - Import the dataset and inspect it.
   - Perform exploratory data analysis.
    - Check the number of movies and reviews over the years.
    - Check the distribution of number of reviews per movie with the exact counting and KDE plot (Kernel Density Estimate).
    - Determine if a class imbalance exists.

2. **Preprocess the Data:**
    - Normalize the review data to remove digits, punctuations marks, etc. and ensure all text is lowercase.

3. **Separate the Train and Test Data into Separate Dataframes**
    - Separate train and test set records into separate dataframes. (The dataset was already split into training and testing sets as denoted by the `ds_part` field.)

4. **Model Training:**
    - Train the following models:
        - dummy classifier
        - NLTK, TF-IDF, and Logistic Regression
        - spaCy, TF-IDF, and Logistic Regression
        - spaCy, TF-IDF, and LGBMClassifier
        - BERT

5. **Model Evaluation:**
    - Test the models for the given test dataset.

6. **Classify My Own Reviews :**
    - Write a few of my own reviews and classify them with each model.

7. **Insights**:
   - Explain the differences in performance among the models for both the given test set and for my own reviews. 

## Requirements <a id="requirements"></a>

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
The `requirements.txt` file includes the following packages:
- `google-colab==1.0.0`
- `tqdm==4.65.0`
- `pandas==1.5.3`
- `numpy==1.23.5`
- `matplotlib==3.7.1`
- `seaborn==0.12.2`
- `nltk==3.8.1`
- `scikit-learn==1.2.2`
- `spacy==3.5.3`
- `lightgbm==3.3.5`
- `transformers==4.31.0`
- `torch==2.0.1`
   

## Results <a id="results"></a>

The predicted probabilities (of a positive class) for my personal reviews were quite different based on the model used. Models 1, 2, and 3 were more similar to one another as a whole than model 4 (BERT), which had less variation in its spread of probabilities and was more neutral in its interpretation of my reviews (aka the probabilities were closer to 0.5). The probabilities of models 1, 2, and 3 seemed to be highly impacted by language with traditionally strong negative or positive connotations and weren't as strong at understanding them in context.

My reviews lost some important context after preprocessing them for the models; they contained mixed sentiments and even a touch of sarcasm, which may explain some of the predictions. My reviews were also very short so there wasn't as much language or context to interpret after they were preprocessed for the models.

The BERT model definitely interpreted the context of my reviews best, but the context was complicated by the normalized text.

Overall, models 1 (NLTK, TF-IDF, and LR) and 2 (spacy, TF-IDF, and Logistic Regression) performed the best with an F1 score of 0.88.
