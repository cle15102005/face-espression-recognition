import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import f1_score

from tqdm import tqdm

class SRC:
    def __init__(self, **kwargs):
        print("(+) Initializing SRC model...")
        
        # Default parameter values.
        params = {
        'n_nonzero_coefs' : 50,         #control sparsity
        'variance_threshold' : 0.98,    #control how much variance PCA should retain  
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params
    
    def create_model(self):
        self.src = OrthogonalMatchingPursuit(n_nonzero_coefs=self.params['n_nonzero_coefs'])
        self.pca = PCA(n_components=self.params['variance_threshold'])
        self.scaler = StandardScaler()
        return self.src, self.pca, self.scaler
    
    def train(self, X_raw, y_train):
        
        self.create_model()
        
        X_scaled = self.scaler.fit_transform(X_raw)
        
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.X_train= normalize(X_pca, axis=1)
        self.y_train = y_train

    def transform(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        return normalize(X_pca, axis=1)

    def predict(self, X_test):
        preds = []
        unique_classes = np.unique(self.y_train)

        for x in tqdm(X_test, desc="Classifying Samples"):
            min_error = float('inf')
            pred_class = None

            for cls in unique_classes:
                A = self.X_train[self.y_train == cls]
                self.src.fit(A.T, x.ravel())
                x_hat = A.T @ self.src.coef_
                error = np.linalg.norm(x - x_hat)

                if error < min_error:
                    min_error = error
                    pred_class = cls

            preds.append(pred_class)
        return preds

    def hyperparameter_tuning(self, X_train_raw, y_train, X_val_raw, y_val):
        print("(+) Starting hyperparameter tuning...")
        best_f1 = -1
        best_params = {}

        nonzeros = [10, 20, 30, 50, 70, 100]
        variances = [0.905, 0.95, 0.98, 0.99, 0.995, 0.9995]
        total = len(nonzeros) * len(variances)

        with tqdm(total=total, desc="Hyperparameter Search") as pbar:
            for nzc in nonzeros:
                for var in variances:
                    self.params['n_nonzero_coefs'] = nzc
                    self.params['variance_threshold'] = var

                    self.train(X_train_raw, y_train)
                    X_val = self.transform(X_val_raw)
                    y_pred = self.predict(X_val)

                    f1 = f1_score(y_val, y_pred, average='macro')

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {'n_nonzero_coefs': nzc, 'variance_threshold': var}

                    pbar.update(1)

        print(f"Best parameters found: {best_params}")
        self.params = best_params  # Update model with best parameters