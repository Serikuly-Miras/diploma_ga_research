import numpy as np
import taichi as ti

from FNN import FNN
from dataclasses import dataclass


@dataclass
class SnakeSettings:
    # snake game specific
    field_size: int
    view_depth: int
    steps_without_apple: int     # number of steps snake can make without food

    # neural network
    nn_neurons_count: list[int]
    nn_activations: list[str]
    nn_params_count: int

    # genetic algorithm
    population_size: int
    epochs: int
    crossover_rate: float
    mutation_rate: float
    elitism_num: float


@ti.data_oriented
class SnakeGA():
    def __init__(self, settings: SnakeSettings) -> None:
        ti.init(kernel_profiler=True)

        # array of snakes bodies
        self.bodies = ti.Vector.field(
            n=2, dtype=int,
            shape=(settings.population_size, settings.field_size**2))

        self.lengths = ti.field(dtype=int, shape=(settings.population_size))

        # food position for each snake
        self.food_positions = ti.Vector.field(
            n=2, dtype=int, shape=(settings.population_size))

        # each snake "score" as value
        self.fitness = ti.field(dtype=int, shape=(settings.population_size))

        # log progress results
        self.history = ti.field(dtype=int, shape=(settings.epochs))

        # snake neural network weights
        self.nn_genes = ti.field(dtype=float, shape=(
            settings.population_size, settings.nn_params_count))

        # genetic algorithm
        # pool for tornament selection
        self.selection_pool = ti.field(
            dtype=int, shape=(settings.population_size))

        # crossover and mutation pool
        self.nn_next_genes = ti.field(dtype=float, shape=(
            settings.population_size, settings.nn_params_count))

        # ti epoch counter
        self.curr_epoch = ti.field(dtype=int, shape=1)

        # shotcuts for self.settings.***
        self.field_size = settings.field_size
        self.view_depth = settings.view_depth
        self.steps_without_apple = settings.steps_without_apple

        self.nn_neurons_count = settings.nn_neurons_count
        self.nn_activations = settings.nn_activations
        self.nn_params_count = settings.nn_params_count

        self.population_size = settings.population_size
        self.epochs = settings.epochs
        self.crossover_rate = settings.crossover_rate
        self.mutation_rate = settings.mutation_rate
        self.elitism_num = settings.elitism_num

        self.place_snakes()
        self.place_food()
        self.ramdomize_genes()

    @ti.kernel
    def place_snakes(self):
        # place all snakes in the middle of the field, reset lenghts
        for i in range(self.population_size):
            self.bodies[i, 0].xy = self.field_size // 2, self.field_size // 2
            self.lengths[i] = 1

    @ti.kernel
    def place_food(self):
        # place food randomly in field (different for each snake)
        for i in range(self.population_size):
            self.food_positions[i].x = ti.random(int) % self.field_size
            self.food_positions[i].y = ti.random(int) % self.field_size

    @ti.kernel
    def ramdomize_genes(self):
        for i in range(self.population_size):
            for j in range(self.nn_params_count):
                self.nn_genes[i, j] = ti.random(float) - 0.5

    # Epochs loop
    def advance(self):
        for i in range(self.epochs):
            self.curr_epoch[0] = i

            # play game for each snake
            for j in range(self.population_size):
                self.fitness[j] = self.calc_fitness(snake_index=j)

            # perform genetic arithm
            self.genetic_arithm()

    def calc_fitness(self, snake_index: int) -> int:
        brain = FNN(neurons_count=self.nn_neurons_count,
                    activations=self.nn_activations, params=self.nn_genes.to_numpy()[snake_index])
        nn_input = np.random.random(size=self.nn_neurons_count[0]) - 0.5
        res = 0
        for i in range(100):
            if np.argmax(brain.predict(input=nn_input)) == 1:  # WIP force one direction moves
                res += 1
        return res

    # GA implementation below

    def genetic_arithm(self):
        self.tornament_selection()
        self.crossover()
        self.mutation()
        self.elitism()
        self.log()
        self.switch_population()

    @ti.kernel
    def tornament_selection(self):
        for i in range(self.population_size):
            left = ti.random(int) % self.population_size
            right = ti.random(int) % self.population_size
            res = left if self.fitness[left] > self.fitness[right] else right
            self.selection_pool[i] = res

    @ti.kernel
    def crossover(self):
        for i in range(int(self.population_size // 2)):
            split = int(ti.random(float) * self.nn_params_count)
            left, right = 0.0, 0.0
            for j in range(self.nn_params_count):
                if (j < split):
                    left = self.nn_genes[self.selection_pool[2 * i], j]
                    right = self.nn_genes[self.selection_pool[2 * i + 1], j]
                else:
                    right = self.nn_genes[self.selection_pool[2 * i], j]
                    left = self.nn_genes[self.selection_pool[2 * i + 1], j]
                self.nn_next_genes[2 * i, j] = left
                self.nn_next_genes[2 * i + 1, j] = right

    @ti.kernel
    def mutation(self):
        for i in range(int(self.mutation_rate * self.population_size * self.nn_params_count)):
            snake = ti.random(int) % self.population_size
            gen = ti.random(int) % self.nn_params_count
            self.nn_next_genes[snake, gen] = ti.random(float)

    @ti.kernel
    def elitism(self):
        pass

    @ti.kernel
    def switch_population(self):
        for i in range(self.population_size):
            for j in range(self.nn_params_count):
                self.nn_genes[i, j] = self.nn_next_genes[i, j]

    @ti.kernel
    def log(self):
        self.history[self.curr_epoch[0]] = self.avg_fit()

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
