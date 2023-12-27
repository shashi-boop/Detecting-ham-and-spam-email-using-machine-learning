from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classifiers():
    train_data = pd.read_csv("train_data.csv")
    X_train, X_val, y_train, y_val = train_test_split(train_data["email"], train_data["label"], test_size=0.2,
                                                      stratify=train_data["label"], random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    accuracy_scores = {}

    for classifier_name, classifier in classifiers.items():
        trained_model = classifier.fit(X_train_vec, y_train)
        acc_score = trained_model.score(X_val_vec, y_val)
        accuracy_scores[classifier_name] = acc_score

    accuracy_values = list(accuracy_scores.values())
    classifier_names = list(accuracy_scores.keys())

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(y=classifier_names, x=accuracy_values, palette="viridis")
    plt.xlabel("Accuracy")
    plt.title("Classifier Performance Comparison")

    # Format x-axis to percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    # Add percentage text on each bar
    for idx, p in enumerate(ax.patches):
        percentage = '{:.1%}'.format(accuracy_values[idx])
        ax.annotate(percentage, (p.get_width() + 0.01, p.get_y() + p.get_height() / 2), ha='left', va='center')

    plt.show()

evaluate_classifiers()
