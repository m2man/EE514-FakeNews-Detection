# EE514 Fake News Detection

This is the repository of EE514 Fake News Detection Assignment 2020. Noted that this project is not aiming to get the highest performance but focussing on explaining the solution as well as exploratory data analysis.

## Requirement
Install the ```requirements.txt```

Download Glove pretrained model [here](https://nlp.stanford.edu/projects/glove/)

Download pretrained model in the assignment [here](https://drive.google.com/file/d/1WYDbmiqCCt5k4-AULkGZRCQYVhOXq0gI/view?usp=sharing)

## Dataset
The dataset is ```fake_news.json``` in which there are 3 columns indicating the headlines, groundtruth if a headline is sarcastic or not, and the URL where a headline came from. However, we did not use the URL column as the assignment requires to predict only based on headlines.

## Exploratory Data Analysis
The ```EDA.ipynb``` shows the EDA stage where some hidden information extracted from the data set such as reliable headlines tend to use determiners more frequent than sarcastic headlines. There are also some boxplots in this notebooks. Based on such finding, some basic features describing headlines are extracted and I performed feature selection by using Pearson correlation coefficient. The library used to extract features is writen in ```myfunctions.py``` script.

## Machine Learning Classifiers
The ```Machine Learning Classifiers.ipynb``` is about applying some basic classifiers (SVM, RF, Logistic Regression, and MLP) with these aforemention extracted features to make the prediction. Noted that the MLP model used in this notebooks is from scikit-learn package, hence it is just a simple neural network. Since the extracting features can take a while to finish, I save them into ```basic_ft.joblib``` files that can be loaded directly for faster running.

Besides, this file also contains the experiments of training SVM, RF, and LR on Bag of Word (BOW) and TFIDF Vectorizer features. Both setting are run with different options of using original headlines or lemmatised headlines. I also perform feature selection by using Logistic Regression with L1 penalty function. Then run all experiments again with the selected features which are saved as ```selected_vocab_s1.joblib```.

## BOW and TFIDF with Deep Learning Classifiers
The ```BOW and TFIDF with Deep Learning Classifiers.ipynb``` is the code for applying MLP with Dropout and Batch Normalisation techniques to mitigate the overfitting issue when the number of features are too much while the number of samples is limited. The models are implemented in PyTorch framework. The supporting functions and classes for running deep learning in PyTorch is stored in ```deep_pytorch.py```.

## Embedding 
The ```Embedding Deep Learning Classifiers.ipynb``` is about learning the Embedding layer to represent words into vectors. A headline now can be converted into the same vector space by taking the mean of all words within itself. The headline then went through a MLP layer to be classifed as real or fake news. The model is implemented in PyTorch. The supporting functions and classes for running deep learning in PyTorch is stored in ```embedding_pytorch.py```.

## Sequence Model
The ```Sequence Models.ipynb``` implements the Bidirectional GRU sequence model to capture the order of words in headlines. Since this model performs best among all, I run the evaluation on test set in this notebooks. ROC and confusion matrix is also reported here. The supporting functions and classes for running deep learning in PyTorch is stored in ```rnn_pytorch.py```.
