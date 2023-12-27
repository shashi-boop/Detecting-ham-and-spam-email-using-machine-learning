import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# creating a function to read  train_data and test_data from previously saved CSV files
def read_data(training_data,testing_data):
    training_data = pd.read_csv(training_data)
    testing_data = pd.read_csv(testing_data)
    return training_data, testing_data

training_data, testing_data = read_data("train_data.csv","test_data.csv")

stpwrds_rmvl = TfidfVectorizer(stop_words='english', max_features=1000, min_df=5)

#creating a function for removing stopwords and infrequent word removal in training and test dataset
def rmval_stpwrds_infrqnt_qrds(train_data,test_data):
    train_email_tfidf = stpwrds_rmvl.fit_transform(train_data['email'])
    test_email_tfidf = stpwrds_rmvl.transform(test_data['email'])
    return train_email_tfidf,test_email_tfidf

train_email_tfidf, test_email_tfidf= rmval_stpwrds_infrqnt_qrds(training_data,testing_data)

def extracting_labels(train_labels,test_labels ):
    # Extracting the labels from train and test dataset
    train_labels = training_data['label']
    test_labels = testing_data['label']
    return train_labels,test_labels

train_labels,test_labels=extracting_labels(training_data,testing_data)

#getting the feature names of the mails
feature_names = stpwrds_rmvl.get_feature_names_out()
# creating a function to print results of feauture selection
def TRAINING_TF_IDF_VALUES():
    print("\n TF-IDF encoding Values for training set first email: ")
    for wrd, vlue in zip(feature_names, train_email_tfidf[0].toarray()[0]):
        if vlue != 0:
            print(f"{wrd}: {vlue}")
def TESTING_TF_IDF_VALUES():
    print("\n TF-IDF encoding values for testing set first email:")
    for wrd, vlue in zip(feature_names, test_email_tfidf[0].toarray()[0]):
        if vlue != 0:
            print(f"{wrd}: {vlue}")

#calling the function
train_results=TRAINING_TF_IDF_VALUES()
test_results=TESTING_TF_IDF_VALUES()


