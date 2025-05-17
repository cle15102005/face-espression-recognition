from sklearn.decomposition import PCA
import sklearn.svm as svm 

class SVM(object):
    def __init__(self, **kwargs):
    
        print("(+) Initializing SVM model...")
        
        # Default parameter values.
        params = {
            'kernel' : 'rbf',
            'C' : 10,
            'gamma' : 'scale',
            'n_components' : 100,
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params
        
    def create_model(self):
        self.pca = PCA(n_components= self.params['n_components'], random_state=18)
        self.svc = svm.SVC(kernel= self.params['kernel'], 
                           C=self.params['C'], 
                           gamma= self.params['gamma'])

    def train(self, X_train, y_train):
        
        self.create_model()
        
        print("(+) PCA dimension reduction...")
        X_train_pca = self.pca.fit_transform(X_train)
        print(f"Dimension after PCA: {X_train_pca.shape[1]}")

        self.svc.fit(X_train_pca, y_train)

    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.svc.predict(X_pca)
