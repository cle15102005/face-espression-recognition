import numpy as np
import data_loader, eval, svm, rf, src
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._omp")

def detect(model_name):
    #load data
    print("(+) Loading data...")
    #train_csv = 'train_face_emotion_named.csv'
    #val_csv = 'val_face_emotion_named.csv'
    train_csv = 'emotional_train_limited.csv'
    val_csv = 'emotional_val_limited.csv'
    X_train_raw, X_val, y_train, y_val = data_loader.load_data(train_csv, val_csv)

    #set default params and create model
    print(f"(+) Creating {model_name} model...")
    if model_name == 'SVM':
        params = {'kernel' : 'rbf', 'C' : 10, 'gamma' : 'scale', 'n_components' : 100}
        detector = svm.SVM(**params)
    elif model_name == 'RF':
        params = {'n_estimators': 100, 'max_depth': None, 'variance_threshold': 0.98, 'random_state': 42}
        detector = rf.RF(**params)
    elif model_name == 'SRC':
        params = {'n_nonzero_coefs' : 30, 'variance_threshold' : 0.95}
        detector = src.SRC(**params)
    
    #tuning option, just for rf and src
    option = input('(?) Enable tuning (Y/n): ').upper()
    if option == "Y":
        idx = np.random.choice(len(X_val), size= 600, replace=False)
        X_val_tune = X_val[idx]
        y_val_tune = y_val[idx]
        params= detector.hyperparameter_tuning(X_train_raw, y_train, X_val_tune, y_val_tune)
    
    #train model
    print(f"(+) Training {model_name} model...")
    detector.train(X_train_raw, y_train)
        
    #recogizing expression
    print("(+) Predicting...")
    y_pred = detector.predict(X_val)

    #evaluation
    eval.evaluate(model_name, y_val, y_pred)

if __name__ == '__main__':
    model_name = input('(?) Enter model name (SVM/RF/SRC): ').upper()
    detect(model_name)