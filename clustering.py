import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple

def run_gmm(X: np.ndarray, n_components: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """
    Fit a Gaussian Mixture Model (GMM) and return:
    - hard cluster labels
    - soft membership probabilities
    - fitted GMM model
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)  # soft probabilities
    return labels, probs, gmm
