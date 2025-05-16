import numpy as np
from src import SRC
import data_loader
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._omp")

def SRC_detect():
    #load data
    print("Loading data...")
    #train_csv = 'train_face_emotion_named.csv'
    #val_csv = 'val_face_emotion_named.csv'
    train_csv = 'emotional_train_limited.csv'
    val_csv = 'emotional_val_limited.csv'
    X_train_raw, X_val_raw, y_train, y_val = data_loader.load_data(train_csv, val_csv)

    #create model
    print("(+) Creating SRC model...")
    params = {
            'n_nonzero_coefs' : 50, 
            'variance_threshold' : 0.98,        
            }
    src = SRC(**params)
    
    option = input('Enable tuning: Y/n: ').upper()
    if option == "Y":
        idx = np.random.choice(len(X_val_raw), size= 900, replace=False)
        X_val_tune = X_val_raw[idx]
        y_val_tune = y_val[idx]
        params= src.hyperparameter_tuning(X_train_raw, y_train, X_val_tune, y_val_tune)
    
    #train model
    print("(+) Training SRC model...")
    src.train(X_train_raw, y_train)

    #transform validation dataset
    print("(+) Transforming validation set...")
    X_val = src.transform(X_val_raw)

    #recogizing expression
    print("(+) Predicting...")
    y_pred = src.predict(X_val)

    #evaluation
    src.evaluate(y_val, y_pred)

if __name__ == '__main__':
    SRC_detect()