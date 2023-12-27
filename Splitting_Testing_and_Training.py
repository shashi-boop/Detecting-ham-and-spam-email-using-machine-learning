import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import string
from sklearn.model_selection import train_test_split




# setting the directories for ham and spam folder
ham_direc='C:\\Users\\shash\\OneDrive\\Desktop\\Applied Machine Learning\\enron1\\enron1\\ham'
spam_direc='C:\\Users\\shash\\OneDrive\\Desktop\\Applied Machine Learning\\enron1\\enron1\\spam'

def loading_emails(ham_direc,spam_direc):
    ham_emails_collection=[]
    for fname in os.listdir(ham_direc):
        with open(os.path.join(ham_direc, fname), 'r') as file:
            ham_emails_collection.append((file.read(), 0))
    spam_emails_collection=[]
    for fname in os.listdir(spam_direc):
        with open(os.path.join(spam_direc, fname), 'r', errors="ignore") as file:
            spam_emails_collection.append((file.read(), 1))
    return ham_emails_collection, spam_emails_collection


# combining all the emails into emails variable
ham_emails_collection, spam_emails_collection = loading_emails(ham_direc, spam_direc)
emails = ham_emails_collection + spam_emails_collection

# creating a DataFrame and storing it into data variable
data = pd.DataFrame(emails, columns=["email", "label"])

# cleaning data and preprocessing
def cleaning_preprocessing(data):
    data = data.drop_duplicates(subset="email")
    data = data[data["email"].str.strip() != ""]

    data["email"] = data["email"].apply(lambda email: (
        BeautifulSoup(email, "html.parser").get_text().lower()
        .translate(str.maketrans("", "", "@"))
        .translate(str.maketrans("", "", "0123456789"))
        .translate(str.maketrans("", "", string.punctuation))
        .replace("\n", " ").replace("\t", " ").replace("\r", " ")
        .strip()
    ))
    return data



# splitting the dataset into training and test
training_data, testing_data = train_test_split(data, test_size=0.3, random_state=42)


# saving the splitting into csv files
training_data.to_csv("train_data.csv")
testing_data.to_csv("test_data.csv")

# calculating the no of ham and spam emails in train and test
train_spam_counts = training_data['label'].value_counts()
test_spam_counts = testing_data['label'].value_counts()

# printing the statistics of trainging and test dataset
def printing_statistics():
    print("Training set statistics:")
    print("Ham emails:", train_spam_counts[0])
    print("Spam emails:", train_spam_counts[1])

    print("\nTest set statistics:")
    print("Ham emails:", test_spam_counts[0])
    print("Spam emails:", test_spam_counts[1])

printing_statistics()


