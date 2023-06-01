# This is the AI & ML project of Fake News Detection.
# -------------------------------------------------------------------------------------------------------------------
# Importing necessary libraries: pandas for data manipulation, 
# numpy for numerical operations, 
# seaborn and matplotlib.pyplot for data visualization.
# -------------------------------------------------------------------------------------------------------------------
# Importing modules from scikit-learn for model selection, accuracy scoring, and generating classification reports.
# Importing modules for regular expressions and string operations.
# -------------------------------------------------------------------------------------------------------------------

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re 
import string 

# Reading the fake news and true news datasets from CSV files using pd.read_csv().
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Displaying the first few rows of the fake news and true news datasets.
data_fake.head()
data_true.head()

# Adding a new column 'class' to both datasets and assigning 0 to fake news and 1 to true news.
data_fake["class"] = 0
data_true['class'] = 1

# Displaying the shape (number of rows and columns) of the fake news and true news datasets.
data_fake.shape, data_true.shape 

# Creating a subset of the last 10 rows from the fake news dataset for manual testing,
# And then removing those rows from the original dataset.
data_fake_manual_testing = data_fake.tail(10)
for i in range (23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)


# Creating a subset of the last 10 rows from the true news dataset for manual testing,
# And then removing those rows from the original dataset.
    data_true_manual_testing = data_true.tail(10)
    for i in range (21416,21406,-1):
        data_true.drop([i], axis = 0, inplace = True)
 

# Assigning the class label 0 (fake news) to the manually tested fake news subset
# And class label 1 (true news) to the manually tested true news subset.
data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

# Displaying the first 10 rows of the manually labeled fake news testing dataset for inspection and analysis
data_fake_manual_testing.head(10)


# Concatenating the fake news and true news datasets vertically to create a merged dataset,
# And displaying the first 10 rows of the merged dataset
data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)

# Displaying the column names of the merged dataset.
data_merge.columns

# Creating a new dataset data by dropping the 'title', 'subject', and 'date' columns from the merged dataset.
data = data_merge.drop(['title','subject', 'date'], axis = 1)

# Checking the number of missing values in each column of the data dataset.
data.isnull().sum()

# Shuffling the rows of the data dataset randomly using the sample function.
data = data.sample(frac = 1)

# Displaying the first 10 rows of the shuffled data dataset.
data.head()

# Resetting the index of the data dataset and removing the old index column.
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)

# Displaying the column names of the data dataset.
data.columns

# Displaying the first 10 rows of the shuffled data dataset AGAIN.
data.head()


# Defining a function wordopt() that performs various text preprocessing operations 
# Using regular expressions and string operations.
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Applying the function wordopt() to the title column of the data dataset.
data['text'] = data['text'].apply(wordopt)

# Assigning the preprocessed 'text' column to x as the input feature 
# And the 'class' column to y as the target variable.
x = data['text']
y = data['class']

# Splitting the preprocessed data into training and testing sets using the train_test_split() function
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

# Importing the TfidfVectorizer class from scikit-learn for text vectorization.
from sklearn.feature_extraction.text import TfidfVectorizer

# Creating an instance of TfidfVectorizer and transforming the training and testing data into TF-IDF vectors.
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Importing the LogisticRegression class from scikit-learn for logistic regression modeling.
from sklearn.linear_model import LogisticRegression

# Creating an instance of LogisticRegression model, fitting it to the training data.
LR = LogisticRegression()
LR.fit(xv_train,y_train)

# Making predictions on the testing data using the trained logistic regression model.
pred_lr=LR.predict(xv_test)

# Calculating the accuracy score of the logistic regression model on the testing data.
LR.score(xv_test, y_test)

# Printing the classification report that includes precision, recall, F1-score,
# and support for each class based on the logistic regression predictions.
print(classification_report(y_test, pred_lr))

# Importing the DecisionTreeClassifier class from scikit-learn for decision tree modeling.
from sklearn.tree import DecisionTreeClassifier

# Creating an instance of DecisionTreeClassifier model, fitting it to the training data.
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# Making predictions on the testing data using the trained decision tree model.
pred_dt = DT.predict(xv_test)

# Calculating the accuracy score of the decision tree model on the testing data.
DT.score(xv_test, y_test)

# Printing the classification report based on the decision tree predictions.
print(classification_report(y_test, pred_dt))


# Importing the GradientBoostingClassifier class from scikit-learn for gradient boosting modeling.
from sklearn.ensemble import GradientBoostingClassifier

# Creating an instance of GradientBoostingClassifier model, fitting it to the training data.
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)


# Making predictions on the testing data using the trained gradient boosting model
pred_gb = GB.predict(xv_test)

# Calculating the accuracy score of the gradient boosting model on the testing data.
GB.score(xv_test, y_test)

# Printing the classification report based on the gradient boosting predictions.
print(classification_report(y_test, pred_gb))


# Importing the RandomForestClassifier class from scikit-learn for random forest modeling.
from sklearn.ensemble import RandomForestClassifier

# Creating an instance of RandomForestClassifier model, fitting it to the training data.
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

# Making predictions on the testing data using the trained random forest model.
pred_rfc = RFC.predict(xv_test)

# Printing the classification report based on the random forest predictions.
print(classification_report(y_test, pred_rfc))


# Defining a function output_lable() that maps the predicted label (0 or 1) to a corresponding text label.
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


# Defining a function manual_testing() that takes a news text as input, preprocesses it using the wordopt()
# function, transforms it into a TF-IDF vector, and makes predictions using all four models
# (logistic regression, decision tree, gradient boosting, and random forest).
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GB[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# Taking user input for a news text, calling the manual_testing() function 
# with the input text, and printing the predictions of all four models for that news text

news = str(input())
manual_testing(news)

