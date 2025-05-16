from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import numpy as np
#plot
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(self, y_true, y_pred):
        # Mapping numeric labels to expression names
        label_to_expression = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }

        # Create ordered label names for consistent plotting and reporting
        classes = sorted(np.unique(y_true))
        target_names = [label_to_expression[i] for i in classes]

        # Evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
