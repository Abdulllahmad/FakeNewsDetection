## Fake News Detection Python Code
## https://github.com/Abdulllahmad/FakeNewsDetection

This Python code is designed to detect fake news articles using machine learning techniques. It leverages several libraries and follows a specific approach for classification. The code provides functionality for training models, evaluating their performance, and performing manual testing.

## Libraries Used :-

The code requires the following libraries to be installed:

    pandas (pip install pandas)
    numpy (pip install numpy)
    seaborn (pip install seaborn)
    matplotlib (pip install matplotlib)
    scikit-learn (pip install scikit-learn)

## Approach Used :-

    Reading Data: The code reads the fake news and true news datasets from CSV files using pd.read_csv() and assigns them to respective variables.

    Data Preprocessing: The code performs various preprocessing steps, such as adding a "class" column, removing unnecessary columns, shuffling the data, and applying text preprocessing operations using regular expressions and string operations.

    Training and Testing Data Split: The preprocessed data is split into training and testing sets using train_test_split() from scikit-learn.

    Text Vectorization: The code utilizes the TfidfVectorizer class from scikit-learn to transform the text data into TF-IDF vectors, which capture the importance of each word in a document.

    Model Training and Evaluation: Four classification models are trained and evaluated on the TF-IDF transformed data:
        Logistic Regression (LR)
        Decision Tree Classifier (DT)
        Gradient Boosting Classifier (GB)
        Random Forest Classifier (RFC)

    For each model, the code trains the model using the training data, makes predictions on the testing data, calculates accuracy scores, and prints classification reports containing precision, recall, F1-score, and support for each class.

    Manual Testing: The code includes a manual_testing() function that allows users to input a news text. The text is preprocessed, transformed into a TF-IDF vector, and passed through all four models for prediction. The code displays the predictions of each model for the provided news text.

## Instructions for .csv Files :-

To use this code, ensure that you have two CSV files: "Fake.csv" and "True.csv." These files should contain the fake news and true news datasets, respectively. Make sure the CSV files have the following columns:

    "text": Contains the text of the news articles.
    "title": Contains the titles of the news articles (used in preprocessing).
    "subject": Contains the subjects/categories of the news articles (not used in the code).
    "date": Contains the dates of the news articles (not used in the code).

Make sure the column names match the code, and the data is properly formatted in the CSV files. Place the CSV files in the same directory as the Python script or provide the correct file paths when using pd.read_csv().

To perform manual testing, follow the prompts in the code and input a news text. The code will preprocess the input, transform it into a TF-IDF vector, and predict the label using all four models. The predicted labels will be displayed for each model.


## Disclaimer :-

This code serves as a basic implementation for fake news detection and may not achieve optimal performance. It is recommended to further refine and enhance the code to improve the accuracy and reliability of the fake news detection system.

Please note that the provided code snippet does not contain the part where the user should save the data into a CSV file.