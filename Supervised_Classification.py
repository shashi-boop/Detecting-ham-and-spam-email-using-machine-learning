import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

train_data = pd.read_csv("train_data.csv")




def partition_dataset(train_data, proportion=0.3, seed=414):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=proportion, random_state=seed)
    for train_indices, validation_indices in stratified_split.split(train_data["email"], train_data["label"]):
        training_emails = train_data["email"].iloc[train_indices]
        validation_emails = train_data["email"].iloc[validation_indices]
        training_labels = train_data["label"].iloc[train_indices]
        validation_labels = train_data["label"].iloc[validation_indices]
    return training_emails, validation_emails, training_labels, validation_labels

training_emails, validation_emails, training_labels, validation_labels = partition_dataset(train_data)


#pipeline for  TfidfVectorizer and Logistic Regression
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=1000, min_df=5)),
    ("classifier", LogisticRegression(solver="liblinear"))
])

#Train the classifier using the pipeline
pipeline.fit(training_emails, training_labels)

#Performing cross validation and printing the mean accuracy
cv_scores = cross_val_score(pipeline, training_emails, training_labels, cv=5)
print(f"C mean accuracy of cross_validation: {cv_scores.mean()}")

#validation accuracy
y_val_pred = pipeline.predict(validation_emails)
val_accuracy = accuracy_score(validation_labels, y_val_pred)
print(f"Validation accuracy: {val_accuracy}")

#classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(validation_labels, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(validation_labels, y_val_pred))

