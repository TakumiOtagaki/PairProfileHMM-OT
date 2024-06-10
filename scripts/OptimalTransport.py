from modules.pairHMM import PairProfileHMM
import numpy as np
import matplotlib.pyplot as plt
import ot
import sys
import pandas as pd

LARGE_NUMBER = 1e20

# main
alphabets = ['A', 'C', 'G', 'T']  # 0, 1, 2, 3
parameters = {  # parameters for PairProfileHMM
    'w_D': 0.1,
    'w_I': 0.1,
    'gamma': 0.3,
    'alpha_D': 0.1,
    'alpha_I': 0.1,
    'beta_D': 0.3,
    'beta_I': 0.3,
    'Match_per_mismatch': 20
}


def OT_on_pairHMM(x_seq, y_seq, parameters, alphabets):
    pairHMM = PairProfileHMM(alphabets=alphabets, **parameters)

    Match_logp = pairHMM.logp_ij_Match_matrix(x_seq, y_seq)  # matrix
    Deletion_logp = pairHMM.logp_ij_Deletion_matrix(
        x_seq, y_seq)  # vector, len = len(x_seq)
    Insertion_logp = pairHMM.logp_ij_Insertion_matrix(
        x_seq, y_seq)  # vector, len = len(y_seq)

    # Optimal Transport
    n1, n2 = len(x_seq), len(y_seq)
    n = n1 + n2
    a = np.ones(n)
    b = np.ones(n)
    a = a / np.sum(a)
    b = b / np.sum(b)
    M = np.zeros((n1 + n2, n1 + n2))
    T = int(n * 0.8)
    M[:n1, :n2] = - Match_logp
    M[n1:, :n2] = - Insertion_logp
    M[:n1, n2:] = - Deletion_logp.T
    # M[n1:, n2:] は -Insertion_lopg と -Deletion_logp の全体で見た最小値にしておく。それは -Match_logp の max よりも大きい
    M[n1:, n2:] = - max(np.min(Insertion_logp), np.min(Deletion_logp))

    M[T:n, T:n] = 0
    if -max(np.min(Insertion_logp), np.min(Deletion_logp)) < -np.max(Match_logp):
        print("Warning: M[n1:, n2:] is smaller than M[:n1, :n2]")
    # M に含まれる - np.inf を 大きな数に変換
    print("M: \n", pd.DataFrame(M))

    # sys.exit()
    M[M == np.inf] = LARGE_NUMBER
    # P = ot.emd(a, b, M)
    P = ot.emd(a, b, M)

    # Plot the transport plan
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot connections based on the transport plan matrix P
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                ax.plot([0, 1], [i, j], lw=P[i, j] * 10,
                        color=plt.cm.viridis(P[i, j] / P.max()))

    # Add labels for the sequences
    for i, char in enumerate(x_seq + "-" * n2):
        ax.text(-0.1, i, char, horizontalalignment='right',
                fontsize=12, color='red')

    for j, char in enumerate(y_seq + "-" * n1):
        ax.text(1.1, j, char, horizontalalignment='left',
                fontsize=12, color='blue')

    # Add color bar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=P.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Transport Amount')

    # Add percentage of transport amount
    for i in range(n):
        for j in range(n):
            if a[i] > 1e-5 and P[i, j] / a[i] > 1e-2:
                mid = (i + j) / 2
                mid += 0.2 * (j - i) + np.sign(j - i) * 0.1
                ax.text(0.5, mid, f'{P[i, j] * 100 / a[i]:.2f}%',
                        horizontalalignment='center', fontsize=10, color='black')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1, max(len(a), len(b)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['x_seq', 'y_seq'])
    ax.set_yticks([])
    ax.set_title('Optimal Transport Plan')
    fig.savefig(
        "/Users/ootagakitakumi/Library/Mobile Documents/com~apple~CloudDocs/大学院/森下研究室/sequence_similarity/figures/optimal_transport.png")
    return P, M


def main():
    # indel について--> 全然ダメ仕方ない。indel をグラフで表現できないようにしているから。
    # x_seq = "ACGTGCA"
    # y_seq = "ACGCA"

    # リピート
    # x_seq = "ACGTACGT"
    # y_seq = "ACGTACGTACGT"

    # indel が交互にはいる
    x_seq = 'ACTACTAGG'
    y_seq = 'ACCTAGATGATGG'

    # ミスマッチ
    # x_seq = "AATGCA"
    # y_seq = "AATGTA"

    # x_seq = "AGTAGTCC"
    # y_seq = "CCAGTCCAGTCC"

    # x_seq = "AGTAGTCCCC"
    # y_seq = "CCAGTCCAGT"

    # x_seq = "TTGAACCCCCCCAGT"
    # y_seq = "AAGTTCCCCCAGT"

    pairHMM = PairProfileHMM(alphabets=alphabets, **parameters)

    P, M = OT_on_pairHMM(x_seq, y_seq, parameters, alphabets)

    print("OT cost = ", np.sum(P * M))


if __name__ == '__main__':
    main()
