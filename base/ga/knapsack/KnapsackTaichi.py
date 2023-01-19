import numpy as np
import taichi as ti

from dataclasses import dataclass


@dataclass
class KnapsackSettings:
    population_size: int
    num_objects: int
    iterations: int
    profits: np.ndarray
    weights: np.ndarray
    capacity: int
    mutation_rate: float
    elitism_num: float


@ti.data_oriented
class KnapsackTaichi:
    def __init__(self, settings: KnapsackSettings, architecture, max_num_threads: int):

        # Init Taichi on CPU with default int and float
        ti.init(arch=architecture, kernel_profiler=True, default_ip=ti.i32, default_fp=ti.f32,
                cpu_max_num_threads=max_num_threads)

        self.s = settings
        pop_shape = (settings.population_size, settings.num_objects)
        self.population = ti.field(ti.u8, pop_shape)
        self.new_gen = ti.field(ti.u8, pop_shape)

        self.selection_pool = ti.field(ti.u32, settings.population_size)

        self.fitness = ti.field(ti.u32, (settings.population_size))
        self.best = ti.field(ti.u32, (settings.iterations))
        self.avg = ti.field(ti.f64, (settings.iterations))
        self.epoch = ti.field(int, (1))

        self.profits = ti.field(int, settings.num_objects)
        self.weights = ti.field(int, settings.num_objects)

        # transfer profits and weights into taichi fields
        for i in range(settings.num_objects):
            self.profits[i] = settings.profits[i]

        for i in range(settings.num_objects):
            self.weights[i] = settings.weights[i]

        # generate creatures genofond
        self.randomize_population()

    @ti.kernel
    def randomize_population(self):
        for i, j in self.population:
            self.population[i, j] = ti.random(int) % 2

    @ti.kernel
    def advance(self):
        while (self.epoch[0] < self.s.iterations):
            self.calc_fitness()
            self.tornament_selection()
            self.crossover()
            self.mutation()
            self.elitism()
            self.log()
            self.switch_population()
            self.epoch[0] += 1

    @ti.func
    def calc_fitness(self):
        for i in range(self.s.population_size):
            fit = 0
            weight = 0
            for j in range(self.s.num_objects):
                if self.population[i, j] == 1:
                    fit += self.profits[j]
                    weight += self.weights[j]

            if weight > self.s.capacity:
                fit = 0

            self.fitness[i] = fit

    @ti.func
    def tornament_selection(self):
        for i in range(self.s.population_size):
            left = ti.random(int) % self.s.population_size
            right = ti.random(int) % self.s.population_size
            res = left if self.fitness[left] > self.fitness[right] else right
            self.selection_pool[i] = res

    @ti.func
    def crossover(self):
        for i in range(self.s.population_size // 2):
            split = int(ti.random(float) * self.s.num_objects)
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

    @ti.func
    def mutation(self):
        for i in range(int(self.s.mutation_rate * self.s.population_size)):
            c = ti.random(int) % self.s.population_size         # creature
            t = ti.random(int) % self.s.num_objects             # triplet
            self.new_gen[c, t] = 1 - self.new_gen[c, t]         # mutate

    @ti.func
    def elitism(self):
        best = 0
        for i in range(self.s.population_size):
            if self.fitness[best] < self.fitness[i]:
                best = i

        for i in range(self.s.elitism_num):
            place = int(ti.random(float) * self.s.population_size)
            for j in range(self.s.num_objects):
                self.new_gen[place, j] = self.population[best, j]

    @ti.func
    def log(self):
        self.best[self.epoch[0]] = self.best_fit()
        self.avg[self.epoch[0]] = self.avg_fit()

    @ti.func
    def switch_population(self):
        for i in range(self.s.population_size):
            for j in range(self.s.num_objects):
                self.population[i, j] = self.new_gen[i, j]

    @ti.func
    def best_fit(self) -> int:
        b = 0
        for i in range(self.s.population_size):
            if b < self.fitness[i]:
                b = self.fitness[i]
        return b

    @ti.func
    def avg_fit(self) -> float:
        s = 0.0
        for i in range(self.s.population_size):
            s += self.fitness[i]
        return s / self.s.population_size
