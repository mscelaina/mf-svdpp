# MySVDpp
from surprise.prediction_algorithms.algo_base import AlgoBase
import numpy as np
from surprise import accuracy
import matplotlib.pyplot as plt


class MySVDpp(AlgoBase):

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 gamma_all=.007, lambda_all=.02,
                 gamma1=None, gamma2=None, gamma3=None,
                 lambda1=None, lambda2=None, lambda3=None,
                 random_state=None,
                 testset=None
                 ):
        self.gamma1 = gamma1 if gamma1 is not None else gamma_all
        self.gamma2 = gamma2 if gamma2 is not None else gamma_all
        self.gamma3 = gamma3 if gamma3 is not None else gamma_all
        self.lambda1 = lambda1 if lambda1 is not None else lambda_all
        self.lambda2 = lambda2 if lambda2 is not None else lambda_all
        self.lambda3 = lambda3 if lambda3 is not None else lambda_all
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.random_state = random_state
        self.testset = testset

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)
        return self

    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        self.BU = np.zeros(trainset.n_users, dtype=np.double)
        self.BI = np.zeros(trainset.n_items, dtype=np.double)
        self.P = rng.normal(self.init_mean, self.init_std_dev,
                            size=(trainset.n_users, self.n_factors))
        self.Q = rng.normal(self.init_mean, self.init_std_dev,
                            size=(trainset.n_items, self.n_factors))
        self.Y = rng.normal(self.init_mean, self.init_std_dev,
                            size=(trainset.n_items, self.n_factors))
        Z = np.zeros(self.n_factors, dtype=np.double)

        g1 = self.gamma1
        g2 = self.gamma2
        g3 = self.gamma3
        l1 = self.lambda1
        l2 = self.lambda2
        l3 = self.lambda3

        max_Iu_length = 0
        for u in range(trainset.n_users):
            max_Iu_length = max(max_Iu_length, len(trainset.ur[u]))
        Iu = [0]*max_Iu_length

        self.RMSE = list()
        self.MAE = list()
        for current_epoch in range(self.n_epochs):
            print(' processing epoch %d' % current_epoch, flush=True)

            for u, i, r in trainset.all_ratings():

                # items rated by u.
                for k, (j, _) in enumerate(trainset.ur[u]):
                    Iu[k] = j
                nu = len(trainset.ur[u])

                nuq = np.sqrt(nu)

                # compute user implicit feedback
                Pu = self.P[u, :].copy()
                Qi = self.Q[i, :].copy()

                Z[:] = 0
                for k in range(nu):
                    Z += self.Y[Iu[k], :]
                Z /= nuq
                Z += Pu

                # compute current error
                err = r - (self.trainset.global_mean +
                           self.BU[u] + self.BI[i] + np.dot(Qi, Z))

                # update biases
                self.BU[u] += g1 * (err - l1 * self.BU[u])
                self.BI[i] += g1 * (err - l1 * self.BI[i])

                # update factors
                self.P[u, :] += g2 * (err * Qi - l2 * Pu)
                self.Q[i, :] += g2 * (err * Z - l2 * Qi)
                nueq = err * Qi / nuq
                for k in range(nu):
                    j = Iu[k]
                    self.Y[j, :] += g3 * (nueq - l3 * self.Y[j, :])

            predictions = self.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=True)
            mae = accuracy.mae(predictions, verbose=True)
            self.RMSE.append(rmse)
            self.MAE.append(mae)
            # print('  err %lf %lf' % (rmse, mae), flush=True)
            print('', flush=True)

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.BU[u]

        if self.trainset.knows_item(i):
            est += self.BI[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            nu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl = (sum(self.Y[j]
                      for (j, _) in self.trainset.ur[u]) / np.sqrt(nu))
            est += np.dot(self.Q[i], self.P[u] + u_impl)

        return est

    def draw(self):
        plt.plot(range(self.n_epochs), self.RMSE)
        plt.title('RMSE')
        # plt.xlabel('Number of Epochs')
        # plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig('./rmse.png')
        plt.clf()

        plt.plot(range(self.n_epochs), self.MAE)
        plt.title('MAE')
        # plt.xlabel('Number of Epochs')
        # plt.ylabel('MAE')
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig('./mae.png')
        plt.clf()
