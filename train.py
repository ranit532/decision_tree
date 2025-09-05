import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_wine
import mlflow
import mlflow.sklearn
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def train():
    # Load wine dataset from sklearn
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Create a pandas DataFrame and save it as a CSV file
    df = pd.DataFrame(data=np.c_[X, y], columns=wine.feature_names + ['target'])
    df.to_csv('wine_dataset.csv', index=False)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set MLflow experiment
    mlflow.set_experiment("Decision Tree Wine Classification")

    # Start an MLflow run
    run_name = f"DecisionTree_Run_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Hyperparameters
        max_depth = 5
        min_samples_split = 2
        
        # Train a Decision Tree Classifier
        dtc = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        dtc.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dtc.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Generate and save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path, "plots")
        
        # Generate and save decision tree plot
        plt.figure(figsize=(20,10))
        plot_tree(dtc, filled=True, feature_names=wine.feature_names, class_names=[str(i) for i in wine.target_names])
        decision_tree_path = "decision_tree.png"
        plt.savefig(decision_tree_path)
        mlflow.log_artifact(decision_tree_path, "plots")

        # Log the model
        mlflow.sklearn.log_model(dtc, "decision_tree_model")

        # Save the model for the FastAPI server
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, f"decision_tree_model_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.joblib")
        joblib.dump(dtc, model_path)

if __name__ == '__main__':
    train()