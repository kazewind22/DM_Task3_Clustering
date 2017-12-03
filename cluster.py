import numpy as np
from time import time

np.random.seed(22)

class Clustering():
    def __init__(self, K, X, weights=None):
        self.K = K
        self.X = X
        self.N = X.shape[0]
        self.mu = None
        self.clusters = None
        if weights != None:
            self.weights = weights
        else:
            self.weights = np.ones(self.N)

    def _init_mu_with_d2(self):
        self.mu = [self.X[np.random.randint(self.N)]]
        while len(self.mu) < self.K:
            self.mu.append(self._imp_sampling())

    def _init_mu_with_random(self):
        idx = np.random.randint(self.N, size=self.K)
        self.mu = [self.X[i] for i in idx]

    def _init_mu_with_mix(self, ratio):
        self.mu = [self.X[np.random.randint(self.N)]]
        n = int(ratio*self.K)
        while len(self.mu) < n:
            self.mu.append(self._imp_sampling())
        idx = np.random.randint(self.N, size=(self.K-n))
        self.mu += [self.X[i] for i in idx]

    def _distance_to_center(self):
        cent = self.mu
        X = self.X
        self.D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])

    def _cluster_data(self):
        cent = self.mu
        X = self.X
        self.best_mu_idx = np.array([np.argmin([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.clusters = [np.where(self.best_mu_idx==i)[0] for i in range(self.K)]

    def _imp_sampling(self):
        self._distance_to_center()
        probs = self.D2 / self.D2.sum()
        cumprobs = probs.cumsum()
        r = np.random.random()
        idx = np.where(cumprobs >= r)[0][0]
        return self.X[idx]


class Coresets(Clustering):
    def construct_coreset(self, M):
        t0 = time()
        self._init_mu_with_d2()
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
        print("Construct coresets from size {N} to {M} with {C} centers time: {T}"
              .format(N=self.N, M=M, C=self.K, T = time() - t0))
        return self.X[samples_idx], 1./M/probs[samples_idx]*self.weights[samples_idx]

class WeightedKmeans(Clustering):
    def _re_center(self):
        K = self.K
        X = self.X
        w = self.weights
        clus = self.clusters
        self.mu = [(np.average(self.X[clus[i]], axis=0, weights=w[clus[i]]))
                   if len(clus[i]) != 0 else self.mu[i] for i in range(K)]

    def _has_converged(self):
        flag = (set([tuple(a) for a in self.mu]) == \
               set([tuple(a) for a in self.oldmu]))
        return flag

    def _compute_loss(self):
        self._distance_to_center()
        return np.dot(self.weights, self.D2) / self.weights.sum()

    def find_centers(self, method="random", ratio=0.5):
        self._init_mu_with_random()
        self.oldmu = self.mu
        t0 = time()
        if method == "random":
            self._init_mu_with_random()
        elif method == "d2":
            self._init_mu_with_d2()
        elif method == "mix":
            self._init_mu_with_mix(ratio)
        else:
            raise Exception("unknow initialization method")
        print("{method} initialize centers time {T}"
              .format(method=method, T=time()-t0))

        print("Updating K-means...")
        for i in range(10):
            if self._has_converged():
                print("Converged!")
                break
            t0 = time()
            self.oldmu = self.mu
            self._cluster_data()
            self._re_center()
            t1 = time()
            print("iter {i} time {T}".format(i=i, T=t1-t0))
            # loss = self._compute_loss()
            # print("loss: {loss}".format(loss=loss))

        return self.mu

def mapper(key, value):
    # key: None
    # value: 2d numpy array of shape (num_ins, num_dims)
    n_split = 5
    num_per_split = value.shape[0] / n_split
    np.random.shuffle(value)
    samples = []
    weights = []
    for i in range(n_split):
        C = Coresets(num_per_split/20, value[i*num_per_split:(i+1)*num_per_split])
        s, w = C.construct_coreset(num_per_split/5)
        samples.append(s)
        weights.append(w)
    samples = np.vstack(samples)
    weights = np.hstack(weights)

    C2 = Coresets(30, samples, weights)
    samples2, weights2 = C2.construct_coreset(200)

    print("Compressed data {S} to {S2}".format(S=value.shape, S2=samples2.shape))
    yield 0, np.hstack((weights2[:,np.newaxis], samples2))

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: concatenation of the values emitted by the mappers

    coresets = values[:, 1:]
    weights = values[:, 0]
    WKmeans = WeightedKmeans(200, coresets, weights)
    # centers = WKmeans.find_centers(method="d2")
    centers = WKmeans.find_centers(method="mix", ratio=0.5)

    yield np.array(centers)
