from dataclasses import dataclass
import numpy as np


@dataclass
class KnapsackSettings:
    population_size: int
    num_objects: int
    iterations: int
    profits: np.ndarray
    weights: np.ndarray
    capacity: int
    crossover_rate: float
    mutation_rate: float
    elitism_num: float


class Knapsack:
    def __init__(self, settings: KnapsackSettings):

        pop_shape = (settings.population_size, settings.num_objects)
        self.population = np.full(dtype=np.int8, shape=pop_shape, fill_value=np.random.randint(
            2, size=pop_shape))
        self.selection_pool = np.ndarray(dtype=np.int8, shape=pop_shape)
        self.new_gen = np.ndarray(dtype=np.int8, shape=pop_shape)

        self.fitness = np.ndarray(
            dtype=np.int32, shape=settings.population_size)
        self.history = np.ndarray(dtype=np.int32, shape=settings.iterations)

        self.population_size = settings.population_size
        self.num_objects = settings.num_objects
        self.iterations = settings.iterations
        self.profits = settings.profits
        self.weights = settings.weights
        self.capacity = settings.capacity
        self.crossover_rate = settings.crossover_rate
        self.mutation_rate = settings.mutation_rate
        self.elitism_num = settings.elitism_num
        self.epoch = 0

    def best_fit(self) -> int:
        max = 0
        for i in range(self.population_size):
            if max < self.fitness[i]:
                max = self.fitness[i]
        return max

    def avg_fit(self) -> int:
        sum = 0
        for i in range(self.population_size):
            sum += self.fitness[i]
        return sum / self.population_size

    def calc_fitness(self):
        for i in range(self.population_size):
            fit = 0
            weight = 0
            for j in range(self.num_objects):
                if self.population[i, j] == 1:
                    fit += self.profits[j]
                    weight += self.weights[j]

            if weight > self.capacity:
                fit = 0

            self.fitness[i] = fit

    def tornament_selection(self):
        for i in range(self.population_size):
            left = np.random.randint(self.population_size)
            right = np.random.randint(self.population_size)
            res = left if self.fitness[left] > self.fitness[right] else right
            for j in range(self.num_objects):
                self.selection_pool[i, j] = self.population[res, j]

    def crossover(self):
        for i in range(int(self.population_size // 2 * self.crossover_rate)):
            split = np.random.randint(self.num_objects)
            left, right = 0, 0
            for j in range(self.num_objects):
                if (j < split):
                    left = self.selection_pool[2 * i, j]
                    right = self.selection_pool[2 * i + 1, j]
                else:
                    right = self.selection_pool[2 * i, j]
                    left = self.selection_pool[2 * i + 1, j]

                self.new_gen[2 * i, j] = left
                self.new_gen[2 * i + 1, j] = right

    def mutation(self):
        for i in range(int(self.mutation_rate * self.population_size * self.num_objects)):
            c = np.random.randint(self.population_size)         # creature
            t = np.random.randint(self.num_objects)             # triplet
            self.new_gen[c, t] = 1 - self.new_gen[c, t]         # flipping bit

    def elitism(self):
        best = 0
        for i in range(self.population_size):
            if self.fitness[best] < self.fitness[i]:
                best = i

        for i in range(self.elitism_num):
            place = np.random.randint(self.population_size)
            for j in range(self.num_objects):
                self.new_gen[place, j] = self.population[best, j]

    def log(self):
        self.history[self.epoch] = self.best_fit()

    def switch_population(self):
        for i in range(self.population_size):
            for j in range(self.num_objects):
                self.population[i, j] = self.new_gen[i, j]

    def advance(self):
        while(self.epoch < self.iterations):
            self.calc_fitness()
            self.tornament_selection()
            self.crossover()
            self.mutation()
            self.elitism()

            self.log()
            self.switch_population()
            self.epoch += 1
