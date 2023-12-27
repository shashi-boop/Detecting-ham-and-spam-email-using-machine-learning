import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd

def build_vectorizer():
    return TfidfVectorizer(stop_words='english')

def assemble_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forests": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Machine": SVC()
    }

def draw_confusion_mat(models, test_data, y_preds):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for ax, model_name, y_test_pred in zip(axs.flatten(), models, y_preds):
        cm = confusion_matrix(test_data["label"], y_test_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

def analyze_models(train_data, test_data):
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(train_data["email"])
    X_test_vec = vectorizer.transform(test_data["email"])

    models = assemble_models()
    y_preds = []
    best_accuracy = 0
    bst_mdel_nme = ""
    best_model = None

    for model_name, model in models.items():
        model.fit(X_train_vec, train_data["label"])
        y_test_pred = model.predict(X_test_vec)
        y_preds.append(y_test_pred)

        accuracy = accuracy_score(test_data["label"], y_test_pred)
        precision = precision_score(test_data["label"], y_test_pred)
        recall = recall_score(test_data["label"], y_test_pred)
        f1 = f1_score(test_data["label"], y_test_pred)

        print(f"{model_name} Metrics:")
        print("Accuracy: {:.2%}".format(accuracy))
        print("Precision: {:.2%}".format(precision))
        print("Recall: {:.2%}".format(recall))
        print("F1 Score: {:.2%}\n".format(f1))

    draw_confusion_mat(models, test_data, y_preds)
    if bst_mdel_nme == "Support Vector Machine":
        with open("best_model_svm.pkl", "wb") as file:
            pickle.dump(best_model, file)
        print(f"Best model '{bst_mdel_nme}' saved as 'best_model_svm.pkl'")

def calling_functions():
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_data.csv")
    analyze_models(train_data, test_data)

calling_functions()

