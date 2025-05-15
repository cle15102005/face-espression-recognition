import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SRC:
    def __init__(self, n_nonzero_coefs=50, variance_threshold=0.98):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.X_train = None
        self.y_train = None

    def create_model(self, X_raw):
        X_scaled = self.scaler.fit_transform(X_raw)
        self.pca = PCA(n_components=self.variance_threshold)
        X_pca = self.pca.fit_transform(X_scaled)
        return normalize(X_pca, axis=1)

    def transform(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        return normalize(X_pca, axis=1)

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def recognize(self, X_test):
        print("Recognizing with SRC...")
        preds = []
        unique_classes = np.unique(self.y_train)

        for x in tqdm(X_test, desc="Classifying Samples"):
            min_error = float('inf')
            pred_class = None

            for cls in tqdm(unique_classes, desc="Evaluating Classes", leave=False):
                A = self.X_train[self.y_train == cls]
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
                omp.fit(A.T, x.ravel())
                x_hat = A.T @ omp.coef_
                error = np.linalg.norm(x - x_hat)

                if error < min_error:
                    min_error = error
                    pred_class = cls

            preds.append(pred_class)
        return preds

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
