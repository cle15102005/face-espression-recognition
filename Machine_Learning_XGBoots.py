import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
df = pd.read_csv("C:/Users/FPT/Downloads/train_face_emotion_named.csv")

# Split into features and labels
X = df.drop(["Image", "label"], axis=1)
y = df["label"]
print(y.value_counts())


# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=100)  # Choose based on variance or trial
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


xgb = XGBClassifier(eval_metric='mlogloss')

grid = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                    scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

predicted_emotions = [emotion_labels[label] for label in y_pred]
true_emotions = [emotion_labels[label] for label in y_test]

# Build final results DataFrame
df_results = pd.DataFrame(X_test)
df_results["True_Label"] = y_test.values
df_results["True_Emotion"] = true_emotions
df_results["Predicted_Label"] = y_pred
df_results["Predicted_Emotion"] = predicted_emotions

print(df_results.head())

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Use previously defined emotion_labels
labels = [emotion_labels[i] for i in sorted(emotion_labels)]

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.tight_layout()
plt.show()
