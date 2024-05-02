# NLP Project: Political Bias Text Classification

**Team Members**:

1. Joseph Allred - joal3275@colorado.edu
   - CONTRIBUTIONS: Scribed meetings, built deliverables, found datasets, explored related works, developed README & all documentation, evaluated a variety of publishers biases.
2. Jevan Wiltz - jewi1870@colorado.edu 
   - CONTRIBUTIONS: Found datasets, explored related works, built first implementations of Naive Bayes, attempted to improve model, developed README & all documentation.
3. Streck Salmon - stsa4539@colorado.edu
   - CONTRIBUTIONS: Found datasets, built first implementation of LSTM.
4. Ben Lipman - beli4539@colorado.edu
   - CONTRIBUTIONS: Contributed to webscraping script for BABE, found datasets, explored RNN implementations.
5. Bowman Russel - boru7277@colorado.edu
   - CONTRIBUTIONS: Contributed to webscraping script for BABE, found datasets, exploring combining datasets for increased model performance.


**Description:**: <br/>
Welcome to our NLP Project! We are group 3 in the Natural Language Processing course CSCI 3832-001 at CU Boulder in the Spring 2024 Semester. In this project, we will delve into the realm of text classification by constructing various models and training them on a dataset comprising news articles. Just as every day inundates us with new information shaping our perspectives, the integrity of this information is paramount. Ensuring its neutrality and lack of bias empowers readers to form uninfluenced and rational conclusions. Our endeavor explores a spectrum of NLP algorithms and machine learning models aimed at discerning political bias within media articles.

**Potential Studies:** <br/>
In addition to our performance analysis, we have a couple different studies we want to conduct
on our data in order to test improvement amongst each individually:

*Cross-Domain Performance Hypothesis/Source Credibility Hypothesis*
- Hypothesis: Models trained on articles from one source platform (e.g., Reddit, general online magazines) may not perform as effectively when tested on articles from a different source (e.g., government websites, news cites, etc).
- Method? Train models using articles from a specific source platform and evaluate their performance when applied to articles sourced from alternative platforms.

*Geopolitical Events Influence Hypothesis*
- Hypothesis: Geopolitical events such as elections, wars, or diplomatic summits will influence the political bias of news articles. (i.e. utilizing headlines to test for bias or not)
- Method? Track major geopolitical events and analyze their correlation with changes in political bias scores of news articles around those events.

*Length of Article Bias Hypothesis*
- Hypothesis: Longer articles tend to exhibit more nuanced and potentially biased language compared to shorter articles.
- Method: Compare the length of articles with their assigned political bias scores to assess the relationship between article length and bias.

*Impact of Data Size Hypothesis*
- Hypothesis: Increasing the size of the training data will improve the performance of the RNN models.
- Method?  Train RNN models using different sizes of training data and observe changes in performance metrics.

*Word Frequency Bias Hypothesis*
- Hypothesis: The frequency of certain politically charged words (e.g., "liberal," "conservative") within articles correlates with the perceived political bias.
- Method? Analyze the occurrence of politically charged words in articles and compare it with the assigned political bias scores.

*Interpretability Hypothesis (Prefer this as last resort if anything)*
- Hypothesis: Naive Bayes models will be more interpretable compared to RNN and LSTM models.
- Method? Analyze the feature importance in Naive Bayes models and compare it to the attention weights in RNN and LSTM models.

### 1. Environment Setup

1. **Python Installation**:
   - If you haven't already installed Python, download and install it from the [official Python website](https://www.python.org/). Ensure you install Python 3.7 or later.

2. **Library Installation**:
   - Open a terminal or command prompt.
   - Install required libraries using pip:
     ```
     pip install tensorflow scikit-learn pandas numpy matplotlib newspaper3k requests beautifulsoup4
     ```

### 2. Dataset

#### Obtaining the Dataset

Our data scraped directives we gathered through a modified scraper we designed, which draws upon the dataset from url https://huggingface.co/datasets/mediabiasgroup/BABE/resolve/main/BABE.csv

You have two options to obtain the dataset:

##### Scraping Data

1. **Run the Scraping Scripts**:
   - Utilize the provided Jupyter notebook (`playing_with_scraping.ipynb`) to scrape news articles from various sources. This notebook contains a modified scraper designed specifically for our project's needs.
   - We collected data from various sources using this scraper, including Fox News, Alternet, and NBC News.

##### Pre-Scraped Data

1. **Download Pre-Scraped Data**:
   - Download the pre-scraped dataset (`BABE_scraped.csv`) from our GitHub repository's `scraped_data` folder.
   - This CSV file contains columns for URL, article content, and political bias type.

### 3. Data Preprocessing

#### Formatting Data

Ensure the scraped data is formatted properly:

1. **Convert Text to Lowercase**:
   - Convert all text data to lowercase to ensure consistency in processing.

2. **Drop Missing Values**:
   - Remove any rows with missing values in the 'content' column.

### 4. Models

#### LSTM (Bidirectional)

This model processes text data using bidirectional LSTM layers to classify political bias.

1. **Load and Tokenize Data**:
   - Load the preprocessed data.
   - Tokenize the text using the `Tokenizer` class from Keras.

2. **Pad Sequences**:
   - Pad sequences to ensure uniform length using `pad_sequences` method from Keras.

3. **Train Model**:
   - Build and train the LSTM model using the provided Jupyter notebook (`test_lstm.ipynb`).
   - Monitor training progress and early stopping using appropriate callbacks.

4. **Evaluate Model**:
   - Evaluate the trained model on the test set.
   - Calculate accuracy and other relevant metrics.
  
#### Simple RNN for Text Classification

This model implemented using PyTorch for text classification tasks. The model utilizes pre-trained word embeddings from GloVe and is trained on a dataset of news articles to classify them into different categories.

1. **Load Data and Preprocess**:
   - Load news articles dataset and preprocess text data.
   - Convert text to lowercase and handle missing values.

2. **Prepare Datasets and DataLoaders**:
   - Split the data into training, validation, and test sets.
   - Create custom PyTorch Dataset and DataLoader objects for efficient training.

3. **Embedding Layer**:
   - Utilize pre-trained word embeddings from GloVe to initialize the embedding layer.
   - Freeze the embedding layer during training to prevent weight updates.

4. **Model Architecture**:
   - Build a Simple RNN model consisting of embedding, RNN, and fully connected layers.
   - Train the model using cross-entropy loss and Adam optimizer.

5. **Training Loop**:
   - Iterate over the training dataset in mini-batches.
   - Update model parameters using backpropagation and monitor validation loss.

6. **Evaluation**:
   - Evaluate the trained model on the test set to assess accuracy.
   - Calculate accuracy metrics to measure model performance.

#### Naive Bayes

This model applies the Naive Bayes algorithm to classify political bias based on text features.

1. **Load and Vectorize Data**:
   - Load the preprocessed data.
   - Vectorize the text data using `TfidfVectorizer` from scikit-learn.

2. **Split Data**:
   - Split the data into training, validation, and testing sets.

3. **Train Model**:
   - Train the Naive Bayes classifier on the training set using the provided Jupyter notebook (`NB_Baseline.ipynb`).

4. **Evaluate Model**:
   - Evaluate the trained model on the test set.
   - Print classification report containing precision, recall, F1-score, and support for each class.

### 5. Running the Code

#### LSTM Model

1. **Execute Script**:
   - Open and run the provided Jupyter notebook `test_lstm.ipynb`.

#### Naive Bayes Model

1. **Execute Script**:
   - Open and run the provided Jupyter notebook `NB_Baseline.ipynb`.
  
#### RNN Model

1. **Execute Script**:
   - Open and run the provided Jupyter notebook `basic_RNN.ipynb`.

## Further Exploration

In addition to our performance analysis, we plan to conduct further studies to test various hypotheses related to political bias in news articles. These hypotheses include cross-domain performance, influence of geopolitical events, article length bias, word frequency bias, and interpretability of models.
