import numpy as np
from rfc import RFC
import dataloader
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def RFC_detect():
    print("Loading data...")
    # train_csv = 'train_face_emotion_named.csv'
    # val_csv = 'val_face_emotion_named.csv'
    train_csv = 'emotional_train_limited.csv'
    val_csv = 'emotional_val_limited.csv'
    X_train_raw, X_val_raw, y_train, y_val = dataloader.load_data(train_csv, val_csv)

    print("(+) Creating RFC model...")
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'variance_threshold': 0.98,
        'random_state': 42
    }
    rfc = RFC(**params)

    option = input('Enable tuning: Y/n: ').upper()
    if option == "Y":
        idx = np.random.choice(len(X_val_raw), size=900, replace=False)
        X_val_tune = X_val_raw[idx]
        y_val_tune = y_val[idx]
        rfc.hyperparameter_tuning(X_train_raw, y_train, X_val_tune, y_val_tune)

    print("(+) Training RFC model...")
    rfc.train(X_train_raw, y_train)

    print("(+) Predicting...")
    y_pred = rfc.predict(X_val_raw)

    print("(+) Evaluating...")
    from sklearn.metrics import classification_report
    print(classification_report(y_val, y_pred))

if __name__ == '__main__':
    RFC_detect()
