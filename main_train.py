import pandas as pd 
from src import SRC
import data_loader

def SRC_detect():
    print("Loading data...")
    train_csv = 'train_face_emotion_named.csv'
    val_csv = 'val_face_emotion_named.csv'
    X_train_raw, X_val_raw, y_train, y_val = data_loader.load_data(train_csv, val_csv)

    print("Initializing SRC...")
    src = SRC(n_nonzero_coefs=50)

    print("Training model...")
    X_train = src.create_model(X_train_raw)
    src.train(X_train, y_train)

    print("Transforming validation set...")
    X_val = src.transform(X_val_raw)

    print("Predicting...")
    y_pred = src.recognize(X_val)

    src.evaluate(y_val, y_pred)

if __name__ == '__main__':
    SRC_detect()