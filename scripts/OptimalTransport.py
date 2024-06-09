from modules.pairHMM import PairProfileHMM
import numpy as np
import matplotlib.pyplot as plt
import ot
from collections import Counter

# main
alphabets = ['A', 'C', 'G', 'T']  # 0, 1, 2, 3
parameters = {  # parameters for PairProfileHMM
    'w_D': 0.05,
    'w_I': 0.05,
    'gamma': 0.5,
    'alpha_D': 0.1,
    'alpha_I': 0.1,
    'beta_D': 0.2,
    'beta_I': 0.2
}


def main():
    # x_seq = "GGTTGGCCCCTTTGGTT"
    # y_seq = "TTGGGTTTTC"

    # x_seq = 'AAACGTTTTGTT'
    # y_seq = 'ACGTACGTT'
    # x_seq = "AATAATAATGAAT"
    # y_seq = "AATAATGAATGAAT"

    # x_seq = "AGTAGTCC"
    # y_seq = "CCAGTCCAGTCC"

    # x_seq = "AGTAGTCCCC"
    # y_seq = "CCAGTCCAGT"

    # x_seq = "TTGAACCCCCCCAGT"
    # y_seq = "AAGTTCCCCCAGT"

    x_seq = "AATAACGTAAT"
    y_seq = "CGTAATAATGCAAT"

    # 長い方を x に、短い方を y に
    x_seq, y_seq = (x_seq, y_seq) if len(
        x_seq) >= len(y_seq) else (y_seq, x_seq)
    # Calculate character frequencies
    x_freq = Counter(x_seq)
    y_freq = Counter(y_seq)

    # Define probability vectors for x and y sequences based on character frequencies
    a = np.array([x_freq[char] / len(x_seq) for char in x_seq])
    b = np.array([y_freq[char] / len(y_seq) for char in y_seq])

    pairHMM = PairProfileHMM(alphabets=alphabets, **parameters)

    logp_ij = pairHMM.logp_ij_matrix(x_seq, y_seq)
    print("logp_ij")
    print(logp_ij)

    # print on graph
    plt.imshow(- logp_ij, cmap='hot', interpolation='nearest')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('- log(P(x_i, y_j)) : distance on OT')
    # 軸に seq を表示
    plt.xticks(np.arange(len(y_seq)), list(y_seq))
    plt.yticks(np.arange(len(x_seq)), list(x_seq))
    plt.colorbar()

    plt.savefig(
        "/Users/ootagakitakumi/Library/Mobile Documents/com~apple~CloudDocs/大学院/森下研究室/sequence_similarity/figures/pairHMM.png")

    # Optimal Transport
    n1, n2 = len(x_seq), len(y_seq)
    a = np.ones(len(x_seq))
    b = np.ones(len(y_seq))
    a = a / np.sum(a)
    b = b / np.sum(b)
    M = - logp_ij
    print(M)
    print(len(a), len(b))
    # M is len(x_seq) x len(y_seq)
    if len(a) > len(b):
        d = len(a) - len(b)
        b = np.concatenate((b, np.zeros(d)), axis=0)
        y_seq += ' ' * d
        M = np.pad(M, ((0, 0), (0, d)), 'constant', constant_values=np.inf)
    elif len(a) < len(b):
        d = len(b) - len(a)
        a = np.concatenate((a, np.zeros(d)), axis=0)
        y_seq += ' ' * d
        M = np.pad(M, ((0, d), (0, 0)), 'constant',
                   constant_values=np.inf)

    # M に含まれる - np.inf を 大きな数に変換
    print(M.shape)
    print(a.shape, b.shape)
    M[M == np.inf] = 1e20
    # P = ot.emd(a, b, M)
    P = ot.emd(a, b, M)
    print(P)
    # EMDistance
    dist = np.sum(P * M)

    # Plot the transport plan
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot connections based on the transport plan matrix P
    for i in range(len(a)):
        for j in range(len(b)):
            if P[i, j] > 0:
                ax.plot([0, 1], [i, j], lw=P[i, j] * 10,
                        color=plt.cm.viridis(P[i, j] / P.max()))

    # Add labels for the sequences
    for i, char in enumerate(x_seq):
        ax.text(-0.1, i, char, horizontalalignment='right',
                fontsize=12, color='red')

    for j, char in enumerate(y_seq):
        ax.text(1.1, j, char, horizontalalignment='left',
                fontsize=12, color='blue')

    # Add color bar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=P.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Transport Amount')
    print("HI")
    # Add percentage of transport amount
    for i in range(n1):
        for j in range(n2):
            print(f"i, j, P[i, j] = {i}, {j}, {P[i, j]}")
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


if __name__ == '__main__':
    main()
