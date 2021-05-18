import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx.generators.community import LFR_benchmark_graph
from networkx.linalg import *
from data_generator import generate_initial_matrix
from community_ga import CommunityGA


def show_matrix(A, title="Matrix"):
    plot, [ax1, ax2] = plt.subplots(1, 2)
    plot.suptitle(title)

    ax1.matshow(A, cmap=plt.cm.Blues)
    '''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            c = A[j, i]
            ax1.text(i, j, str(c), va='center', ha='center')
    '''

    nx.draw(nx.from_numpy_matrix(A), node_size=10)


def main():
    node_count = 14
    matrix = generate_initial_matrix(node_count)

    shuffled_genome = list(range(0, node_count))
    random.shuffle(shuffled_genome)

    GA = CommunityGA(1000, shuffled_genome, matrix)
    matrix = GA.rearrange_matrix(shuffled_genome)
    GA.set_matrix(matrix)

    best_genome = GA.fit(100)
    show_matrix(matrix, "Starting Matrix")
    show_matrix(GA.rearrange_matrix(best_genome), "Final Matrix")
    plt.show()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
