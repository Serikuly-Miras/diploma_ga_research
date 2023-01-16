import numpy as np
import taichi as ti
import sys

from FNN import FNN
from dataclasses import dataclass


@dataclass
class SnakeSettings:
    # snake game specific
    field_size: int
    view_depth: int
    steps_without_food: int     # number of steps snake can make without food

    # neural network
    hidden_topology: list[int]
    hidden_activations: list[str]

    # genetic algorithm
    population_size: int
    epochs: int
    mutation_rate: float
    elitism_num: int

    # auto_save
    auto_save_span: int


@ti.data_oriented
class SnakeGA():
    def __init__(self, settings: SnakeSettings) -> None:
        ti.init(kernel_profiler=True)

        # Snake game
        self.bodies = ti.Vector.field(n=2, dtype=int, shape=(
            settings.population_size, settings.field_size**2))
        self.lengths = ti.field(dtype=int, shape=(settings.population_size))
        self.food_positions = ti.Vector.field(
            n=2, dtype=int, shape=(settings.population_size))
        self.apples_eaten = ti.field(
            dtype=int, shape=(settings.population_size))

        # neural network
        self.input_neurons_count = 8 + (settings.view_depth*2+1)**2 - 1

        self.nn_topology = settings.hidden_topology
        self.nn_topology.insert(0, self.input_neurons_count)
        self.nn_topology.append(4)

        self.nn_activations = settings.hidden_activations
        self.nn_activations.insert(0, 'input')
        self.nn_activations.append('sigmoid')

        self.genes_length = 0
        # dense layers
        for i in range(len(self.nn_topology) - 1):
            self.genes_length += self.nn_topology[i] * self.nn_topology[i + 1]
        # biases
        for i in range(len(self.nn_topology) - 1):
            self.genes_length += self.nn_topology[i+1]

        print("Input neurons count - ", self.input_neurons_count)
        print("NN params count - ", self.genes_length)

        # Genetic algorithm
        self.fitness = ti.field(dtype=float, shape=(settings.population_size))
        self.selection_pool = ti.field(
            dtype=int, shape=(settings.population_size))
        self.next_genes = ti.field(dtype=float, shape=(
            settings.population_size, self.genes_length))
        self.genes = ti.field(dtype=float, shape=(
            settings.population_size, self.genes_length))

        # log progress results
        self.curr_epoch = ti.field(dtype=int, shape=1)
        self.avg = ti.field(dtype=float, shape=(settings.epochs))
        self.best = ti.field(dtype=float, shape=(settings.epochs))
        self.apples = ti.field(dtype=int, shape=(settings.epochs))

        # shotcuts for self.settings.***
        self.field_size = settings.field_size
        self.view_depth = settings.view_depth
        self.steps_without_food = settings.steps_without_food

        self.population_size = settings.population_size
        self.epochs = settings.epochs
        self.mutation_rate = settings.mutation_rate
        self.elitism_num = settings.elitism_num
        self.auto_save_span = settings.auto_save_span

        self.place_food()
        self.randomize_genes()

    @ti.kernel
    def place_food(self):
        # place food randomly in field (different for each snake)
        for i in range(self.population_size):
            self.food_positions[i].x = ti.random(int) % self.field_size
            self.food_positions[i].y = ti.random(int) % self.field_size

    @ti.kernel
    def randomize_genes(self):
        for i in range(self.population_size):
            for j in range(self.genes_length):
                self.genes[i, j] = ti.random(float) * 2 - 1

    # Epochs loop
    def advance(self):
        for i in range(self.epochs):
            self.curr_epoch[0] = i
            sys.stdout.write('\r')
            sys.stdout.write("Epoch %d/%d" % (i+1, self.epochs))
            sys.stdout.flush()

            # play game for each snake
            for j in range(self.population_size):
                self.fitness[j] = self.calc_fitness(snake_index=j)

            # perform genetic arithm
            self.genetic_arithm()

            # save progress
            if (i % self.auto_save_span == 0):
                with open('saves/epoch_{}.npy'.format(i), 'wb+') as f:
                    np.save(f, self.genes.to_numpy())

    def calc_fitness(self, snake_index: int) -> float:
        brain = FNN(neurons_count=self.nn_topology,
                    activations=self.nn_activations, params=self.genes.to_numpy()[snake_index])

        mid = self.field_size//2
        self.bodies[snake_index, 0].xy = mid, mid

        apples = 0
        steps = 0
        move = 0
        steps_left = self.steps_without_food
        while (steps_left > 0):
            nn_input = np.zeros(shape=self.input_neurons_count)
            nn_input[move] = 1
            self.prep_vision(nn_input=nn_input, sn=snake_index)
            move = int(np.argmax(brain.predict(input=nn_input)))

            res = self.validate_game(move=move, sn=snake_index)
            if (res == 0):
                break

            # if apple reset steps left
            if (self.bodies[snake_index, 0].x == self.food_positions[snake_index].x
                    and self.bodies[snake_index, 0].y == self.food_positions[snake_index].y):
                self.food_positions[snake_index] = np.random.randint(
                    self.field_size), np.random.randint(self.field_size)
                steps_left = self.steps_without_food
                apples += 1

            steps_left -= 1
            steps += 1
        self.apples_eaten[snake_index] = apples
        return steps + ti.pow(2, apples) + ti.pow(apples, 2.1) * 500 - ti.pow(apples, 1.2)*ti.pow(steps/4, 1.3)

    @ti.kernel
    def prep_vision(self, nn_input: ti.types.ndarray(), sn: int):
        # 0-7 neurons (tail + head direction)

        # vision neurons
        vision_index = 8
        for i in range(-self.view_depth, self.view_depth+1):
            for j in range(-self.view_depth, self.view_depth+1):
                if (i == 0 and j == 0):
                    continue

                vx = self.bodies[sn, 0].x + i
                vy = self.bodies[sn, 0].y + j

                if (self.food_positions[sn].x == vx and self.food_positions[sn].y == vy):
                    nn_input[vision_index] = 1
                elif (vx < 0 or vx > self.field_size or vy < 0 or vy > self.field_size):
                    nn_input[vision_index] = -1
                else:
                    nn_input[vision_index] = 0

                vision_index += 1

    @ti.kernel
    def validate_game(self, move: int, sn: int) -> int:
        survived = 1

        # move
        if (move == 0):
            self.bodies[sn, 0].x += 1
        elif (move == 1):
            self.bodies[sn, 0].y -= 1
        elif (move == 2):
            self.bodies[sn, 0].x -= 1
        else:
            self.bodies[sn, 0].y += 1

        # validate move
        if (self.bodies[sn, 0].x < 0 or self.bodies[sn, 0].x > self.field_size):
            survived = 0

        if (self.bodies[sn, 0].y < 0 or self.bodies[sn, 0].y > self.field_size):
            survived = 0

        return survived

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
            for j in range(self.genes_length):
                l = self.genes[self.selection_pool[2 * i], j]
                r = self.genes[self.selection_pool[2 * i + 1], j]
                self.next_genes[2 * i, j] = (l + r) / 2
                self.next_genes[2 * i + 1, j] = (l + r) / 2

    @ti.kernel
    def mutation(self):
        for i in range(int(self.mutation_rate * self.population_size)):
            snake = ti.random(int) % self.population_size
            gen = ti.random(int) % self.genes_length
            self.next_genes[snake, gen] += (2*ti.random(float) - 1) / 100.0

    @ti.kernel
    def elitism(self):
        max_index = 0
        for i in range(self.population_size):
            if self.fitness[i] > self.fitness[max_index]:
                max_index = i

        for i in range(self.elitism_num):
            for j in range(self.genes_length):
                self.next_genes[i, j] = self.genes[max_index, j]

    @ti.kernel
    def switch_population(self):
        for i in range(self.population_size):
            for j in range(self.genes_length):
                self.genes[i, j] = self.next_genes[i, j]

    @ti.kernel
    def log(self):
        self.best[self.curr_epoch[0]] = self.best_fit()
        self.avg[self.curr_epoch[0]] = self.avg_fit()
        self.apples[self.curr_epoch[0]] = self.max_apples()

    @ti.func
    def best_fit(self) -> float:
        max_fit = 0
        for i in range(self.population_size):
            if max_fit < self.fitness[i]:
                max_fit = self.fitness[i]
        return max_fit

    @ti.func
    def avg_fit(self) -> float:
        fit_sum = 0.0
        for i in range(self.population_size):
            fit_sum += self.fitness[i]
        return fit_sum / self.population_size

    @ti.func
    def max_apples(self) -> float:
        max_score = 0
        for i in range(self.population_size):
            if max_score < self.apples_eaten[i]:
                max_score = self.apples_eaten[i]
        return max_score
