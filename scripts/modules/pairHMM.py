import matplotlib.pyplot as plt
import numpy as np

import sys

import pandas as pd


class PairProfileHMM:
    def __init__(self,  alphabets, w_D, w_I, gamma, alpha_D, alpha_I, beta_D, beta_I, Match_per_mismatch=3):
        self.alphabets = alphabets
        if 1 - (alpha_D + alpha_I + gamma) <= 0:

            raise ValueError(
                'Sum of alpha_D, alpha_I and gamma must be less than 1')
        if max(w_D, w_I, gamma, alpha_D, alpha_I, beta_D, beta_I) > 1 or min(w_D, w_I, gamma, alpha_D, alpha_I, beta_D, beta_I) < 0:
            raise ValueError('All parameters must be between 0 and 1')

        self.w_D = w_D
        self.w_I = w_I
        self.gamma = gamma
        self.alpha_D = alpha_D
        self.alpha_I = alpha_I
        self.beta_D = beta_D
        self.beta_I = beta_I

        self.DeletionStates = {0, 3, 5}
        self.InsertionStates = {1, 4, 6}
        self.MatchStates = {2}
        self.states_num = 7

        # transition matrix
        self.A = self.get_transition_matrix(w_D, w_I, gamma,
                                            alpha_D, alpha_I, beta_D, beta_I)
        print(self.A)

        self.pi = np.zeros(self.states_num + 1)
        self.pi[0] = self.w_D
        self.pi[1] = 1 - self.w_D
        self.pi[2] = (1 - self.w_I) * self.gamma
        self.pi[3] = (1 - self.w_I) * self.alpha_D
        self.pi[4] = (1 - self.w_I) * self.alpha_I
        self.pi[5] = (1 - self.w_I) * \
            (1 - self.alpha_D - self.alpha_I - self.gamma)
        self.pi[6] = (1 - self.w_I) * (1 - self.alpha_D -
                                       self.alpha_I - self.gamma) * self.w_I

        # emission matrix
        num_alphabets = len(alphabets)
        self.B = np.ones((self.states_num, num_alphabets, num_alphabets))
        # 対角成分だけ大きくする
        self.B += np.diag(np.ones(num_alphabets) * (Match_per_mismatch - 1))
        self.B /= np.sum(self.B, axis=2)[:, :, None]

    def get_transition_matrix(self, w_D, w_I, gamma, alpha_D, alpha_I, beta_D, beta_I):
        transition_matrix = np.zeros(
            (self.states_num + 1, self.states_num + 1))
        transition_matrix[0, 0] = w_D
        transition_matrix[0, 1] = 1 - w_D
        transition_matrix[1, 1] = w_I
        transition_matrix[1, 2] = (1 - w_I) * gamma
        transition_matrix[1, 3] = (1 - w_I) * alpha_D
        transition_matrix[1, 4] = (1 - w_I) * alpha_I
        transition_matrix[1, 5] = (1 - w_I) * \
            (1 - alpha_D - alpha_I - gamma)
        transition_matrix[2, 2] = gamma
        transition_matrix[2, 3] = alpha_D
        transition_matrix[2, 4] = alpha_I
        transition_matrix[2, 5] = 1 - alpha_D - alpha_I - gamma
        transition_matrix[3, 2] = (1 - beta_D) * gamma
        transition_matrix[3, 3] = beta_D
        transition_matrix[3, 4] = (1 - beta_D) * alpha_I
        transition_matrix[3, 5] = (1 - beta_D) * \
            (1 - alpha_D - alpha_I - gamma)
        transition_matrix[4, 2] = (1 - beta_I) * gamma
        transition_matrix[4, 3] = (1 - beta_I) * alpha_D
        transition_matrix[4, 4] = beta_I
        transition_matrix[4, 5] = (1 - beta_I) * \
            (1 - alpha_D - alpha_I - gamma)
        transition_matrix[5, 5] = w_D
        transition_matrix[5, 6] = 1 - w_D
        transition_matrix[6, 6] = w_I
        transition_matrix[6, 7] = 1 - w_I
        transition_matrix[7, 7] = 1
        return transition_matrix

    def convert_alphabet2int(self, x):
        ret = [0] * len(x)
        for i in range(len(x)):
            ret[i] = self.alphabets.index(x[i])
        print(ret)
        return ret

    def forward(self, x_seq, y_seq):
        # x is the 1st sequence which consists of alphabets
        # y is the 2nd sequence which consists of alphabets
        x = [""] + self.convert_alphabet2int(x_seq)  # 1 origin
        y = [""] + self.convert_alphabet2int(y_seq)  # 1 origin

        # initialize forward matrix
        k = self.states_num  # num_of_states
        n_1, n_2 = len(x_seq), len(y_seq)
        forward = np.zeros((k, n_1 + 1, n_2 + 1))
        # s = scaling factor
        s = np.ones((n_1 + 1, n_2 + 1))

        # initialize forward matrix
        for i in range(k):
            forward[i, 0, 0] = self.pi[i]
        for t_1 in range(1, n_1 + 1):  # t_1 = 1, 2, ..., n_1
            for i in range(k):
                forward[i, t_1, 0] = sum([forward[j, t_1 - 1, 0] * self.A[j, i]
                                          * sum(self.B[i, x[t_1], :]) for j in self.InsertionStates])
            s[t_1, 0] = sum(forward[:, t_1, 0])
            forward[:, t_1, 0] /= s[t_1, 0]

        for t_2 in range(1, n_2 + 1):
            for i in range(k):
                forward[i, 0, t_2] = sum([forward[j, 0, t_2 - 1] * self.A[j, i] * sum(self.B[i, :, y[t_2]])
                                         for j in self.DeletionStates])
            s[0, t_2] = sum(forward[:, 0, t_2])
            forward[:, 0, t_2] /= s[0, t_2]

        # main loop
        for t_1 in range(1, n_1 + 1):
            for t_2 in range(1, n_2 + 1):
                for i in range(k):
                    m_ = [forward[j, t_1 - 1, t_2 - 1] *
                          self.A[j, i] * self.B[i, x[t_1], y[t_2]] for j in self.MatchStates]
                    d_ = [forward[j, t_1 - 1, t_2] *
                          self.A[j, i] * sum(self.B[i, x[t_1], :]) for j in self.DeletionStates]
                    i_ = [forward[j, t_1, t_2 - 1] *
                          self.A[j, i] * sum(self.B[i, :, y[t_2]]) for j in self.InsertionStates]
                    # print("m_", m_)

                    forward[i, t_1, t_2] += sum(m_) + sum(d_) + sum(i_)

                s[t_1, t_2] = sum(forward[:, t_1, t_2])
                forward[:, t_1, t_2] /= s[t_1, t_2]
        # forward[k, n1+1, n2+1] だったけど、forward[k, n1, n2] だけ取り出す
        return forward[:, 1:, 1:], s

    def backward(self, x_seq, y_seq, s):
        # x is the 1st sequence which consists of alphabets
        # y is the 2nd sequence which consists of alphabets
        n_1, n_2 = len(x_seq), len(y_seq)
        x = [""] + self.convert_alphabet2int(x_seq)
        y = [""] + self.convert_alphabet2int(y_seq)

        # initialize forward matrix
        k = self.states_num

        backward = np.zeros((k, n_1 + 1, n_2 + 1))

        # initialize backward matrix
        for i in range(k):
            backward[i, n_1, n_2] = 1
        for t_1 in range(n_1 - 1, -1, -1):  # t_1 = n_1 - 1, n_1 - 2, ..., 1
            for i in self.InsertionStates:
                backward[i, t_1, n_2] = sum([backward[j, t_1 + 1, n_2] * self.A[i, j]
                                             * sum(self.B[j, x[t_1 + 1], :]) for j in self.InsertionStates])
            backward[:, t_1, n_2] /= s[t_1 + 1, n_2]

        for t_2 in range(n_2 - 1, -1, -1):
            for i in self.DeletionStates:
                backward[i, n_1, t_2] = sum([backward[j, n_1, t_2 + 1] * self.A[i, j]
                                             * sum(self.B[j, :, y[t_2 + 1]]) for j in self.DeletionStates])
            backward[:, n_1, t_2] /= s[n_1, t_2 + 1]

        # main loop
        for t_1 in range(n_1 - 1, -1, -1):
            for t_2 in range(n_2 - 1, -1, -1):
                for i in range(k):
                    # この辺自信ない。
                    m_ = [backward[j, t_1 + 1, t_2 + 1] * self.A[j, i] *
                          self.B[i, x[t_1 + 1], y[t_2 + 1]] for j in self.MatchStates]
                    d_ = [backward[j, t_1 + 1, t_2] * self.A[j, i] *
                          sum(self.B[i, x[t_1 + 1], :]) for j in self.DeletionStates]
                    i_ = [backward[j, t_1, t_2 + 1] * self.A[j, i] *
                          sum(self.B[i, :, y[t_2 + 1]]) for j in self.InsertionStates]
                    backward[i, t_1, t_2] += sum(m_) + sum(d_) + sum(i_)
                backward[:, t_1, t_2] /= s[t_1 + 1, t_2 + 1]
        return backward[:, :-1, :-1]

    def forward_backward(self, x_seq, y_seq):
        forward, s = self.forward(x_seq, y_seq)
        backward = self.backward(x_seq, y_seq, s)
        return forward, backward, s

    def logp_ij_Match_matrix(self, x_seq, y_seq):
        # x_i, y_j がマッチする確率を求める
        logP = np.zeros((len(x_seq), len(y_seq)))
        f, b, s = self.forward_backward(x_seq, y_seq)

        for i in range(len(x_seq)):
            for j in range(len(y_seq)):
                logP[i, j] = (np.log(f[2, i, j]) + np.log(b[2, i, j]))
        return logP

    def logp_ij_Deletion_matrix(self, x_seq, y_seq):
        # x_i, y_j がマッチする確率を求める
        logP = np.zeros(len(x_seq))
        f, b, s = self.forward_backward(x_seq, y_seq)
        log_px = np.sum(np.log(s)).sum()
        print("log_px: ", log_px)

        for i in range(len(x_seq)):
            for j in range(len(y_seq)):
                for k in self.DeletionStates:
                    logP[i] += np.log(f[k, i, j]) + np.log(b[k, i, j])
        logP -= log_px
        return logP / len(x_seq)

    def logp_ij_Insertion_matrix(self, x_seq, y_seq):
        # x_i, y_j がマッチする確率を求める
        P = np.zeros(len(y_seq))
        f, b, s = self.forward_backward(x_seq, y_seq)
        log_px = np.sum(np.log(s)).sum()

        for j in range(len(y_seq)):
            for i in range(len(x_seq)):
                for k in self.InsertionStates:
                    P[j] += np.log(f[k, i, j]) + np.log(b[k, i, j])
        P -= log_px

        return P / len(y_seq)
