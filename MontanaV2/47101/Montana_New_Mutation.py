import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from numpy import genfromtxt
import copy
import multiprocessing as mp
import time


# Checked
def weights_init(weights):
    new_weights = []
    for w in weights:
        new_weights.append(np.random.laplace(0, 1, w.shape))
    return [new_weights[i] for i in range(len(new_weights))]


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


def define_single_model(neurons):
    model = Sequential()
    model.add(Dense(neurons[1], activation='relu', use_bias=False, input_shape=(neurons[0],)))
    model.add(Dense(neurons[2], activation='relu', use_bias=False))
    model.add(Dense(neurons[3], activation='sigmoid', use_bias=False))
    model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['accuracy'])
    return model


# Checked
def crossover_nodes(population, neurons, weights):
    model = define_single_model(neurons)
    for i in range(0, len(population) - 1, 2):
        a = [weights[z].shape for z in range(0, len(weights))]
        m = [np.empty(a[z]) for z in range(0, len(a))]
        m1 = copy.deepcopy(population[i].get_weights())
        m2 = copy.deepcopy(population[i + 1].get_weights())
        for j in range(0, len(neurons) - 1):
            for h in range(0, neurons[j + 1]):
                if np.random.uniform(0, 1) < 0.5:
                    for k in range(0, len(m1[j])):
                        # print('parent 1 selected and weight is:', m1[j][k][h])
                        m[j][k][h] = copy.deepcopy(m1[j][k][h])
                else:
                    for k in range(0, len(m2[j])):
                        # print('parent 2 selected and weight is:', m2[j][k][h])
                        m[j][k][h] = copy.deepcopy(m2[j][k][h])
        model.set_weights(m)
    return model


# Double Checked
def mutate_nodes(population, neurons, number_of_mutation):
    model = define_single_model(neurons)
    m1 = population.get_weights()
    m = copy.deepcopy(m1)
    for neurons_number in range(neurons[0], np.sum(neurons)):
        if np.random.uniform(0, 1) < 1/(np.sum(neurons) - neurons[0]):
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
            for k in range(0, len(m1[layers_number])):
                # This is working
                m[layers_number][k][neurons_number] = m1[layers_number][k][neurons_number] + random_number
            model.set_weights(m)
    return model


def evaluation(population, x, y):
    # result of evaluation weights network
    evaluation_result = []
    for p in population:
        evaluation_result.append(round(p.evaluate(x, y, verbose=0)[1], 5))
    return evaluation_result


# Checked
def probability(fitness):
    return [round(x / np.sum(fitness), 5) for x in fitness]


# Checked
def breeding(population, neurons, dicts, num, weights):
    parent = []
    if np.random.uniform(0, 1) < 0.5:
        while len(parent) < 1:
            for p in population:
                if dicts[p][1] > np.random.uniform(0, 1) and len(parent) < 1:
                    parent.append(p)
        lit_child = mutate_nodes(parent[0], neurons, num)
    else:
        while len(parent) < 2:
            for p in population:
                if dicts[p][1] > np.random.uniform(0, 1) and len(parent) < 2:
                    parent.append(p)
        lit_child = crossover_nodes(parent, neurons, weights)
    return lit_child


def child_val(child, x_test, y_test, dicts, population):
    fit_result = []
    for p in population:
        fit_result.append(dicts[p][0])
    print('The mean of fitness is:', np.mean(fit_result))
    print('The best of fitness is:', np.max(fit_result))
    child_fit = round(child.evaluate(x_test, y_test, verbose=0)[1], 5)
    child_prob = round(child_fit / (child_fit + np.sum(fit_result) - np.min(fit_result)), 5)
    return [child_fit, child_prob]


def discard_individual(dicts, key):
    del dicts[key]
    return dicts


def insert_child(dicts, key, values):
    dicts[key] = values
    return dicts


def multiprocess(pool, population):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x_test, y_test) for p in population))
    return evaluation_result


x = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(0,))

y_test = y[:40000]
x_test = x[:40000]

y_train = y[40000:50000]
x_train = x[40000:50000]

generations = 100  # Number of generation algorithm run.
pop_size = 50
neurons = [4, 7, 10, 1]
number_of_mutate_node = 2

population = define_model(population_size=pop_size, neurons=neurons)
sample_weights = population[0].get_weights()

fit_result = evaluation(population, x_test, y_test)
# pool = mp.Pool(4)
# fit_result = multiprocess(pool, population)
prob_pop = probability(fit_result)
dicts = {population[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(population))}

# for p in population:
#     if dicts[p][0] == np.max(fit_result):
#         best_GA = copy.deepcopy(p)

for g in range(0, generations):
    print('The generation number is: ', g + 1)
    child = breeding(list(dicts.keys()), neurons, dicts, number_of_mutate_node, sample_weights)
    child_values = child_val(child, x_test, y_test, dicts, list(dicts.keys()))

    # Discarding
    for p in list(dicts.keys()):
        if dicts[p][0] == np.min(fit_result):
            discard_individual(dicts, p)
    print(child_values)
    dicts = insert_child(dicts, child, child_values)
    # Updating Probabilities in Dictionary
    for p in list(dicts.keys()):
        dicts[p][1] = round(dicts[p][0] / np.sum([dicts[p][0] for p in list(dicts.keys())]), 5)

# best_GA.fit(x_train, y_train, epochs=1, validation_split=0.5)