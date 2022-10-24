import numpy as np
import taichi as ti

from Knapsack import KnapsackSettings

@ti.data_oriented
class KnapsackTaichi:
    def __init__(self, settings: KnapsackSettings, architecture, max_num_threads: int):

        # Init Taichi on CPU with default int and float
        ti.init(arch=architecture, kernel_profiler=True, default_ip=ti.i32, default_fp=ti.f32,
                cpu_max_num_threads=max_num_threads)

        pop_shape = (settings.population_size, settings.num_objects)
        self.population = ti.field(ti.u8, pop_shape, order='ij')
        self.selection_pool = ti.field(ti.u8, pop_shape, order='ij')
        self.new_gen = ti.field(ti.u8, pop_shape, order='ij')

        self.fitness = ti.field(int, (settings.population_size))
        self.history = ti.field(int, (settings.iterations))
        self.epoch = ti.field(int, (1))

        self.profits = ti.field(int, settings.num_objects)
        self.weights = ti.field(int, settings.num_objects)

        self.population_size = settings.population_size
        self.num_objects = settings.num_objects
        self.iterations = settings.iterations
        self.capacity = settings.capacity
        self.crossover_rate = settings.crossover_rate
        self.mutation_rate = settings.mutation_rate
        self.elitism_num = settings.elitism_num

        # transfer profits and weights into taichi filds
        for i in range(settings.num_objects):
            self.profits[i] = settings.profits[i]

        for i in range(settings.num_objects):
            self.weights[i] = settings.weights[i]

        # generate creatures genofond
        self.randomize_population()

    @ti.kernel
    def randomize_population(self):
        for i in range(self.population_size):
            for j in range(self.num_objects):
                self.population[i, j] = ti.random(int) % 2

    @ti.func
    def best_fit(self) -> int:
        max = 0
        for i in range(self.population_size):
            if max < self.fitness[i]:
                max = self.fitness[i]
        return max

    @ti.func
    def avg_fit(self) -> int:
        sum = 0
        for i in range(self.population_size):
            sum += self.fitness[i]
        return sum / self.population_size

    @ti.func
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

    @ti.func
    def tornament_selection(self):
        for i in range(self.population_size):
            left = ti.random(int) % self.population_size
            right = ti.random(int) % self.population_size
            res = left if self.fitness[left] > self.fitness[right] else right
            for j in range(self.num_objects):
                self.selection_pool[i, j] = self.population[res, j]

    @ti.func
    def crossover(self):
        for i in range(int(self.population_size // 2 * self.crossover_rate)):
            split = int(ti.random(float) * self.num_objects)
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

    @ti.func
    def mutation(self):
        for i in range(int(self.mutation_rate * self.population_size * self.num_objects)):
            c = ti.random(int) % self.population_size                # creature
            t = ti.random(int) % self.num_objects                    # triplet
            self.new_gen[c, t] = 1 - self.new_gen[c, t]         # flipping bit

    @ti.func
    def elitism(self):
        best = 0
        for i in range(self.population_size):
            if self.fitness[best] < self.fitness[i]:
                best = i

        for i in range(self.elitism_num):
            place = int(ti.random(float) * self.population_size)
            for j in range(self.num_objects):
                self.new_gen[place, j] = self.population[best, j]

    @ti.func
    def log(self):
        self.history[self.epoch[0]] = self.best_fit()

    @ti.func
    def switch_population(self):
        for i in range(self.population_size):
            for j in range(self.num_objects):
                self.population[i, j] = self.new_gen[i, j]

    @ti.kernel
    def advance(self):
        while(self.epoch[0] < self.iterations):
            self.calc_fitness()
            self.tornament_selection()
            self.crossover()
            self.mutation()
            self.elitism()

            self.log()
            self.switch_population()
            self.epoch[0] += 1
