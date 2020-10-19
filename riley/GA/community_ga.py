from genetic_algorithm import GeneticAlgorithm
import numpy as np
import random


class CommunityGA(GeneticAlgorithm):
    def __init__(self, population_size, dna, matrix):
        super().__init__(population_size, dna)
        self._population = []
        for index in range(0, population_size):
            self._population.append(self._mutate(self._base_dna))

        self._matrix = matrix

    def _mutate(self, strand):
        start_node = random.randint(0, len(strand) - 2)
        end_node = random.randint(1, len(strand) - 1)

        individual = np.copy(strand)

        individual[start_node:end_node] = individual[start_node:end_node][::-1] # invert a sublist of the dna

        return individual

    def _evolve(self):
        for index in range(0, int(self._population_size / 2)):
            self._population.append(self._mutate(self._base_dna))

        self._sort_fitness()

    def _evaluate(self, individual):
        bandwidths = []
        temp_matrix = self.rearrange_matrix(individual)
        for row in range(0, self._matrix.shape[0]):
            for col in range(0, self._matrix.shape[1]):
                if temp_matrix[row, col] != 0:
                    bandwidths.append(abs(row-col))

        return max(bandwidths)

    def _sort_fitness(self):
        self._population = sorted(self._population, key=self._evaluate)

        middle = int(len(self._population) / 2)
        self._population = self._population[:middle]

    def fit(self, generations):
        self._sort_fitness()
        for generation in range(generations):
            for individual in range(int(self._population_size / 2)):
                self._population.append(self._mutate(self._population[individual]))

            self._sort_fitness()

        return self._population[0]

    def rearrange_matrix(self, dna):
        row_matrix = np.zeros(self._matrix.shape)
        temp_matrix = np.zeros(self._matrix.shape)

        iterator = 0
        for row in dna:
            row_matrix[:, iterator] = self._matrix[:, row]
            iterator += 1
        iterator = 0
        for col in dna:
            temp_matrix[iterator, :] = row_matrix[col, :]
            iterator += 1

        return temp_matrix

    def set_matrix(self, new_matrix):
        self._matrix = new_matrix



