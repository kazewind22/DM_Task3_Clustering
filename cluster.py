import numpy as np

np.random.seed(22)
class Clustering():
    def __init__(self, K, X):
        self.K = K
        self.X = X
        self.N = X.shape[0]
        self.mu = None

    def _cluster_data(self):
        cent = self.mu
        X = self.X
        self.best_mu_idx = np.array([np.argmin([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.clusters = [np.where(self.best_mu_idx==i)[0] for i in range(self.K)]


class Coresets(Clustering):
    def _distance_to_center(self):
        cent = self.mu
        X = self.X
        self.D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])

    def _imp_sampling(self):
        self._distance_to_center()
        probs = self.D2 / self.D2.sum()
        cumprobs = probs.cumsum()
        r = np.random.random()
        idx = np.where(cumprobs >= r)[0][0]
        return self.X[idx]

    def init_seeding(self):
        self.mu = [self.X[np.random.randint(self.N)]]
        while len(self.mu) < self.K:
            self.mu.append(self._imp_sampling())

    def construct_coreset(self, M):
        self.init_seeding()
        self._distance_to_center()
        self._cluster_data()
        alpha = 16*(np.log(self.K)+2)
        c_phi = self.D2.sum()/self.N
        D2_b = [self.D2[g].sum() for g in self.clusters]
        Size_b = [len(g) for g in self.clusters]
        S = np.array([alpha * self.D2[i] / c_phi +
                      2 * alpha * D2_b[self.best_mu_idx[i]] / Size_b[self.best_mu_idx[i]] / c_phi +
                      4 * self.N / Size_b[self.best_mu_idx[i]] for i in range(self.N)])
        probs = S / S.sum()
        cumprobs = probs.cumsum()
        samples_idx = [np.where(cumprobs >= np.random.random())[0][0] for i in range(M)]
        return self.X[samples_idx], 1./M/probs[samples_idx]

class WeightedKmeans(Clustering):
    def __init__(self, K, X, weights=None):
        self.K = K
        self.X = X
        self.N = X.shape[0]
        self.mu = None
        self.clusters = None
        if weights != None:
            self.weights = weights
        else:
            self.weights = np.ones(K)

    def _init_mu_with_d2(self):
        C = Coresets(self.K, self.X)
        C.init_seeding()
        self.mu = C.mu

    def _init_mu_with_random(self):
        idx = np.random.randint(self.N, size=self.K)
        self.mu = [self.X[i] for i in idx]

    def _re_center(self):
        K = self.K
        X = self.X
        w = self.weights
        clus = self.clusters
        self.mu = [(np.average(self.X[clus[i]], axis=0, weights=w[clus[i]]))
                   if len(clus[i]) != 0 else self.mu[i] for i in range(K)]
        print(len(self.mu))

    def _has_converged(self):
        return(set([tuple(a) for a in self.mu]) == \
               set([tuple(a) for a in self.oldmu]))

    def find_centers(self, method="random"):
        self._init_mu_with_random()
        self.oldmu = self.mu
        if method == "random":
            self._init_mu_with_random()
        elif method == "d2":
            self._init_mu_with_d2()
        else:
            raise Exception("unknow initialization method")
        for i in range(10):
            print("Updating K-means\nIter: {i}".format(i=i))
            if self._has_converged():
                break
            self.oldmu = self.mu
            self._cluster_data()
            self._re_center()
        return self.mu

def mapper(key, value):
    # key: None
    # value: 2d numpy array of shape (num_ins, num_dims)
    C = Coresets(50, value)
    samples, weights = C.construct_coreset(300)
    print("Compressed data {S} to {S2}".format(S=value.shape, S2=samples.shape))
    yield 0, np.hstack((weights[:,np.newaxis], samples))

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: concatenation of the values emitted by the mappers
    # Note that we do *not* output a (key, value) pair here.
    coresets = values[:, 1:]
    weights = values[:, 0]
    WKmeans = WeightedKmeans(200, coresets, weights=weights)
    centers = WKmeans.find_centers(method="random")
    np.array(centers).shape

    yield np.array(centers)
