from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm

class XGB(object):
    def __init__(self, **kwargs):
        print("(+) Initializing XGBoost model...")

        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'variance_threshold': 0.98,
            'random_state': 42
        }

        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        self.model = XGBClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            eval_metric='mlogloss',
            random_state=self.params['random_state']
        )
        self.pca = PCA(n_components=self.params['variance_threshold'])
        self.scaler = StandardScaler()

    def train(self, X_raw, y_train):
        self.create_model()
        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca = self.pca.fit_transform(X_scaled)

        self.X_train = X_pca
        self.y_train = y_train

        self.model.fit(self.X_train, self.y_train)

    def transform(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def predict(self, X_raw):
        X = self.transform(X_raw)
        return self.model.predict(X)

    def hyperparameter_tuning(self, X_train_raw, y_train, X_val_raw, y_val):
        print("(+) Starting hyperparameter tuning...")
        best_f1 = -1
        best_params = {}

        n_estimators = [100, 200]
        max_depths = [4, 6, 8]
        learning_rates = [0.01, 0.1, 0.3]
        variances = [0.95, 0.98, 0.99]
        total = len(n_estimators) * len(max_depths) * len(learning_rates) * len(variances)

        with tqdm(total=total, desc="Hyperparameter Search") as pbar:
            for n_est in n_estimators:
                for depth in max_depths:
                    for lr in learning_rates:
                        for var in variances:
                            self.params['n_estimators'] = n_est
                            self.params['max_depth'] = depth
                            self.params['learning_rate'] = lr
                            self.params['variance_threshold'] = var

                            self.train(X_train_raw, y_train)
                            y_pred = self.predict(X_val_raw)

                            f1 = f1_score(y_val, y_pred, average='macro')

                            if f1 > best_f1:
                                best_f1 = f1
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'variance_threshold': var
                                }

                            pbar.update(1)

        print(f"Best Parameters Found: {best_params}")
        self.params = best_params  # Update the best
