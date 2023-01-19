import numpy as np

from KnapsackTaichi import KnapsackSettings


class Knapsack:
    def __init__(self, settings: KnapsackSettings):
        self.s = settings
        pop_shape = (settings.population_size, settings.num_objects)
        self.population = np.full(dtype=np.int8, shape=pop_shape, fill_value=np.random.randint(
            2, size=pop_shape))
        self.new_gen = np.ndarray(dtype=np.int8, shape=pop_shape)
        self.selection_pool = np.ndarray(
            dtype=int, shape=settings.population_size)
        self.fitness = np.ndarray(
            dtype=np.int32, shape=settings.population_size)
        self.best = np.ndarray(dtype=np.int32, shape=settings.iterations)
        self.avg = np.ndarray(dtype=np.int32, shape=settings.iterations)

        self.epoch = 0

    def advance(self):
        while (self.epoch < self.s.iterations):
            self.calc_fitness()
            self.tornament_selection()
            self.crossover()
            self.mutation()
            self.elitism()

            self.log()
            self.switch_population()
            self.epoch += 1

    def calc_fitness(self):
        for i in range(self.s.population_size):
            fit = 0
            weight = 0
            for j in range(self.s.num_objects):
                if self.population[i, j] == 1:
                    fit += self.s.profits[j]
                    weight += self.s.weights[j]

            if weight > self.s.capacity:
                fit = 0

            self.fitness[i] = fit

    def tornament_selection(self):
        for i in range(self.s.population_size):
            left = np.random.randint(self.s.population_size)
            right = np.random.randint(self.s.population_size)
            res = left if self.fitness[left] > self.fitness[right] else right
            self.selection_pool[i] = res

    def crossover(self):
        for i in range(self.s.population_size // 2):
            split = np.random.randint(self.s.num_objects)
            left, right = 0, 0
            l = self.selection_pool[2 * i]
            r = self.selection_pool[2 * i + 1]

            for j in range(self.s.num_objects):
                if (j < split):
                    left = self.population[l, j]
                    right = self.population[r, j]
                else:
                    left = self.population[r, j]
                    right = self.population[l, j]

                self.new_gen[2 * i, j] = left
                self.new_gen[2 * i + 1, j] = right

    def mutation(self):
        for i in range(int(self.s.mutation_rate * self.s.population_size)):
            c = np.random.randint(self.s.population_size)         # creature
            t = np.random.randint(self.s.num_objects)             # triplet
            self.new_gen[c, t] = 1 - self.new_gen[c, t]           # mutate

    def elitism(self):
        best = 0
        for i in range(self.s.population_size):
            if self.fitness[best] < self.fitness[i]:
                best = i

        for i in range(self.s.elitism_num):
            place = np.random.randint(self.s.population_size)
            for j in range(self.s.num_objects):
                self.new_gen[place, j] = self.population[best, j]

    def switch_population(self):
        for i in range(self.s.population_size):
            for j in range(self.s.num_objects):
                self.population[i, j] = self.new_gen[i, j]

    def log(self):
        self.best[self.epoch] = self.best_fit()
        self.avg[self.epoch] = self.avg_fit()

    def best_fit(self) -> int:
        b = 0
        for i in range(self.s.population_size):
            if b < self.fitness[i]:
                b = self.fitness[i]
        return b

    def avg_fit(self) -> float:
        s = 0.0
        for i in range(self.s.population_size):
            s += self.fitness[i]
        return s / self.s.population_size
