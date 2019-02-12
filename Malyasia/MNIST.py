import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from numpy import genfromtxt
import copy
import multiprocessing as mp
import time
from keras.datasets import mnist
import keras


# Checked
def define_model(population_size, neurons):
    models = []
    for i in range(0, population_size):
        model = Sequential()
        model.add(Dense(neurons[1], activation='relu', use_bias=False, input_shape=(neurons[0],)))
        model.add(Dense(neurons[2], activation='relu', use_bias=False))
        model.add(Dense(neurons[3], activation='sigmoid', use_bias=False))
        model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['accuracy'])
        models.append(model)
    return models


# Checked
def define_single_model(neurons):
    model = Sequential()
    model.add(Dense(neurons[1], activation='relu', use_bias=False, input_shape=(neurons[0],)))
    model.add(Dense(neurons[2], activation='relu', use_bias=False))
    model.add(Dense(neurons[3], activation='sigmoid', use_bias=False))
    model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['accuracy'])
    return model


# Checked
def crossover_nodes_make2child(population, neurons, weights):
    temp = []
    a = [weights[z].shape for z in range(0, len(weights))]
    for i in range(0, len(population) - 1, 2):
        models = define_model(2, neurons)
        nm1 = [np.empty(a[z]) for z in range(0, len(a))]
        nm2 = [np.empty(a[z]) for z in range(0, len(a))]
        m1 = copy.deepcopy(population[i].get_weights())
        m2 = copy.deepcopy(population[i + 1].get_weights())
        for j in range(0, len(neurons) - 1):
            for h in range(0, neurons[j + 1]):
                if np.random.uniform(0, 1) < 0.5:
                    for k in range(0, len(m1[j])):
                        # print('parent 1 selected and weight is:', m1[j][k][h])
                        nm1[j][k][h] = copy.deepcopy(m1[j][k][h])
                        nm2[j][k][h] = copy.deepcopy(m2[j][k][h])
                else:
                    for k in range(0, len(m2[j])):
                        # print('parent 2 selected and weight is:', m2[j][k][h])
                        nm1[j][k][h] = copy.deepcopy(m2[j][k][h])
                        nm2[j][k][h] = copy.deepcopy(m1[j][k][h])
        models[0].set_weights(nm1)
        models[1].set_weights(nm2)
        temp.append(models[0])
        temp.append(models[1])
        del models, m1, m2, nm1, nm2, a
    return temp


def mutate_nodes(population, neurons, number_of_mutation):
    m = population.get_weights()
    for i in range(0, number_of_mutation):
        neurons_number = int(np.random.uniform(neurons[0], np.sum(neurons) - 1))
        layers_number = 0
        # print(neurons_number)
        first = neurons[0]
        last = neurons[0]
        for j in range(0, len(neurons) - 1):
            last += neurons[j + 1]
            # print(first, last)
            if first <= neurons_number < last:
                layers_number = j
                neurons_number = neurons_number - first
                break
            first = last
        # print(layers_number, neurons_number)
        random_number = np.random.normal(0, 0.5, 1)
        for k in range(0, len(m[layers_number])):
            # This is working
            m[layers_number][k][neurons_number] = m[layers_number][k][neurons_number] + random_number
    population.set_weights(m)
    del m
    return population


# Checked !!!!#Sort kon hatman#!!!!
def selecting_for_pool(population, dicts):
    parent = []
    while len(parent) < len(population):
        for p in population:
            if dicts[p][1] > np.random.uniform(0, 1) and len(parent) < len(population):
                parent.append(p)
    return parent


# Either we make sure that no two excact same parents get paired or
# if it happens that two same parents get paired, crossover shuld not happen (cause their children will be the same and it would be a waste of time
# and resource to allow additional wasteful crossover to happen, so the repeated parent must be copied twice into the premilienery
# population that mighr undergoo mutation)
def mating_pool(population, neurons, dicts, weights):
    paired_parents = []
    parents = selecting_for_pool(population, dicts)
    while len(paired_parents) < len(parents):
        t = np.random.choice(parents, 2, False)
        paired_parents.append(t[0])
        paired_parents.append(t[1])
    lit_child = crossover_nodes_make2child(paired_parents, neurons, weights)
    return lit_child

# !!!!#Sort kon hatman#!!!!
def mutate_pool(population, neurons, num):
    for i in range(len(population)):
        if np.random.uniform(0, 1) < (1/len(population)):
            new = mutate_nodes(population[i], neurons, num)
            population[i] = new
            break
    return population


# Checked
def evaluation(population, x, y):
    # result of evaluation weights network
    evaluation_result = []
    for p in population:
        evaluation_result.append(round(p.evaluate(x, y, verbose=0)[1], 5))
    return evaluation_result


# Checked
def probability(fitness):
    return [round(x / np.sum(fitness), 5) for x in fitness]


# Checked and Order is correct
def multiprocess(pool, population, x_test, y_test):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x_test, y_test) for p in population))
    return [e[1] for e in evaluation_result]


pool = mp.Pool(4)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

generations = 1000  # Number of generation algorithm run.
pop_size = 50
neurons = [784, 512, 512, 10]
number_of_mutate_node = 180
number_of_child_generate = 1

population = define_model(population_size=pop_size, neurons=neurons)
sample_weights = population[0].get_weights()

# fit_result = evaluation(population, x_test, y_test)
fit_result = multiprocess(pool, population)

prob_pop = probability(fit_result)
dicts = {population[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(population))}

for g in range(0, generations):
    print('The generation number is: ', g + 1)
    children = mating_pool(list(dicts.keys()), neurons, dicts, sample_weights)
    final_pop = mutate_pool(children, neurons, number_of_mutate_node)

    dicts.clear()

    # fit_result = evaluation(final_pop, x_test, y_test)
    fit_result = multiprocess(pool, final_pop, x_test, y_test)
    prob_pop = probability(fit_result)
    dicts = {population[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(population))}
