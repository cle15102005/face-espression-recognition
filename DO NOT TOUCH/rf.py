import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

class RF(object):
    def __init__(self, **kwargs):
        print("(+) Initializing Random Forest model...")

        params = {
            'n_estimators': 100,
            'max_depth': None,
            'variance_threshold': 0.98,
            'random_state': 42
        }

        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        self.rf = RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            random_state=self.params['random_state']
        )
        self.pca = PCA(n_components=self.params['variance_threshold'])
        self.scaler = StandardScaler()
        return self.rf, self.pca, self.scaler

    def train(self, X_raw, y_train):
        self.create_model()

        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca = self.pca.fit_transform(X_scaled)

        self.X_train = X_pca
        self.y_train = y_train

        self.rf.fit(self.X_train, self.y_train)

    def transform(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def predict(self, X_test_raw):
        X_test = self.transform(X_test_raw)
        return self.rf.predict(X_test)

    def hyperparameter_tuning(self, X_train_raw, y_train, X_val_raw, y_val):
        print("(+) Starting hyperparameter tuning...")
        best_f1 = -1
        best_params = {}

        estimators = [50, 100, 200]
        depths = [None, 10, 20]
        variances = [0.95, 0.98, 0.99]
        total = len(estimators) * len(depths) * len(variances)

        with tqdm(total=total, desc="Hyperparameter Search") as pbar:
            for n_est in estimators:
                for depth in depths:
                    for var in variances:
                        self.params['n_estimators'] = n_est
                        self.params['max_depth'] = depth
                        self.params['variance_threshold'] = var

                        self.train(X_train_raw, y_train)
                        y_pred = self.predict(X_val_raw)

                        f1 = f1_score(y_val, y_pred, average='macro')

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'variance_threshold': var
                            }

                        pbar.update(1)

        print(f"Best Parameters Found: {best_params}")
        self.params = best_params