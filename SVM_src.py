import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class SVMEmotionModel:
    def __init__(self, n_components=100, kernel='rbf', C=10, gamma='scale'):
        self.pca = PCA(n_components=n_components, random_state=18)
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def fit(self, X_train, y_train):
        print("\n💡 Đang giảm chiều dữ liệu bằng PCA...")
        X_train_pca = self.pca.fit_transform(X_train)
        print(f"✔️ Số chiều sau PCA: {X_train_pca.shape[1]}")

        print("🔍 Đang huấn luyện SVM...")
        self.model.fit(X_train_pca, y_train)

    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.model.predict(X_pca)

    def evaluate(self, y_true, y_pred, labels=None):
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))

        print("🔎 Precision (macro):", precision_score(y_true, y_pred, average='macro'))
        print("🔎 Recall (macro):   ", recall_score(y_true, y_pred, average='macro'))
        print("🔎 F1-score (macro):", f1_score(y_true, y_pred, average='macro'))
        print("🔎 F1-score (weighted):", f1_score(y_true, y_pred, average='weighted'))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - SVM with PCA')
        plt.tight_layout()
        plt.show()
