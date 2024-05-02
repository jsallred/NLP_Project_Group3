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


### 1. Environment Setup

1. **Python Installation**:
   - If you haven't already installed Python, download and install it from the [official Python website](https://www.python.org/). Ensure you install Python 3.10 or later.

2. **Library Installation**:
   - Open a terminal or command prompt.
   - Install required libraries using pip:
     ```
     pip install -r requirements.txt
     ```
      or 

     ```
     pip install tensorflow scikit-learn pandas numpy matplotlib newspaper3k requests beautifulsoup4
     ```

### 2. Datasets

#### Obtaining the Training/Testing Dataset

1. By running the 4th block of our scrape_datasets.ipynb we will obtain the raw BABE dataset used to train and test our models.

#### Obtaining the Dataset Used to Evaluate Different News Organization's Biases

1. By running the 2th block of our scrape_datasets.ipynb we will obtain the raw PoliticalBias dataset used evaluate the biases of the top 5 news organizations in the dataset. 

#### Getting the final Unproccessed Dataset (getting web content from the URLs downloaded) 

1. By running the 3rd and 6th blocks of scrape_datasets.ipynb you can obtain the unproccessed final PoliticalBias and BABE datasets respectively. After proccessing these will then be used to train and test our models as well as evaluate the bias of different news organizations.
2. These datasets will be in the .csv form of [url, text_content, bias_label]. Bias label in the form: Left Leaning = 0, Right Leaning = 2, Center = 1.
3. These unproccessed final datasets will later have their text formatted for training, testing and making predictions. 
4. Please note that scrape_datasets.ipynb took our team 6 hours to run with our computational recources. 

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

   **Potential Studies:** <br/>
   
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


## Results

