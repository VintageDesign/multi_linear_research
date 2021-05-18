from abc import ABC

class GeneticAlgorithm(ABC):
    def __init__(self, population_size, dna):
        """
        Initializes the GA with a given population size and dna
        :param population_size: the size of the population
        :param dna: The base genome for each candidate to base their dna off of.
        """

        self._population_size = population_size
        self._base_dna = dna

    def _sort_fitness(self):
        raise NotImplementedError

    def _mutate(self, strand):
        raise NotImplementedError

    def _evolve(self):
        raise NotImplementedError

    def _evaluate(self, inidvidual):
        raise NotImplementedError

    def fit(self, generations):
        raise NotImplementedError
