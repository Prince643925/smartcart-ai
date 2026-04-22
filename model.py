from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(X):
    best_k = 2
    best_score = -1

    for k in range(2, 10):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            if len(set(labels)) > 1:  # avoid invalid silhouette
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
        except:
            continue

    return best_k, best_score


def run_clustering(X):
    best_k, score = find_best_k(X)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels, best_k, score
    
# Runs KMeans clustering & Returns cluster labels