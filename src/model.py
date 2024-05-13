from MulticoreTSNE import MulticoreTSNE
from sklearn.mixture import GaussianMixture


class GMM_MODEL(GaussianMixture):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X):
        self.labels_ = self.fit_predict(X)
        return self

    def predict(self, X):
        return X

class TSNE:
    def __init__(self,n_components=2, random_state=666, **kwargs):
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, embedding=None):
        return None
    
    def transform(self, X):
        self.tsne = MulticoreTSNE(n_components=self.n_components, random_state = self.random_state, n_jobs=-1, **self.kwargs)
        return self.tsne.fit_transform(X)
    
    def fit_transform(self, X):
        return self.transform(X)

if __name__ == "__main__":
    from pprint import pprint

    gm = GMM_MODEL(n_components=10)
    pprint(vars(gm))
    pprint("done")
