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


# Checked
def crossover_nodes(population, neurons):
    p = copy.deepcopy(population[0])
    temp = population[0].get_weights()
    for i in range(0, len(population) - 1, 2):
        a = [temp[z].shape for z in range(0, len(temp))]
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
        p.set_weights(m)
    return p


# Checked
def mutate_nodes(population, neurons, number_of_mutation):
    p = copy.deepcopy(population)
    m1 = p.get_weights()
    m = copy.deepcopy(m1)
    for i in range(0, number_of_mutation):
        neurons_number = int(np.random.uniform(neurons[0], np.sum(neurons) - 1))
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
        random_number = np.random.laplace(0, 1, 1)
        for k in range(0, len(m1[layers_number])):
            m[layers_number][k][neurons_number] = m1[layers_number][k][neurons_number] + random_number
    p.set_weights(m)
    return p


def evaluation(population, x, y):
    # result of evaluation weights network
    evaluation_result = []
    for p in population:
        evaluation_result.append(round(p.evaluate(x, y, verbose=0)[1], 5))
    return evaluation_result


#Still Not Working
def threadpool_evaluation(population, x, y):
    # result of evaluation weights network
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor() as executor:
        evaluation_result = [executor.submit(p.evaluate, x, y, verbose=0) for p in population]
        evaluation_result = [future.result() for future in as_completed(evaluation_result)]

    return evaluation_result


# Checked
def fitness(evaluation_result):
    return [round(x) for x in evaluation_result]


# Multiprocessing fitness
def multiprocess_fitness(evaluation_result):
    return [round(1 / x[0], 5) for x in evaluation_result]


# Checked
def selection(fitness):
    return [round(x / np.sum(fitness), 5) for x in fitness]


# Checked
def breeding(population, neurons, dicts, num):
    parent = []
    if np.random.uniform(0, 1) < 0.5:
        while len(parent) < 1:
            for p in population:
                if dicts[p][2] > np.random.uniform(0, 1) and len(parent) < 1:
                    parent.append(p)
        lit_child = mutate_nodes(parent[0], neurons, num)
    else:
        while len(parent) < 2:
            for p in population:
                if dicts[p][2] > np.random.uniform(0, 1) and len(parent) < 2:
                    parent.append(p)
        lit_child = crossover_nodes(parent, neurons)
    return lit_child


def child_val(child, x_test, y_test, dicts, population):
    fit_result = []
    for p in population:
        fit_result.append(dicts[p][1])
    print('The mean of fitness is:', np.mean(fit_result))
    print('The best of fitness is:', np.max(fit_result))
    child_eval = round(child.evaluate(x_test, y_test, verbose=0)[1], 5)
    child_fit = round(child_eval, 5)
    child_prob = round(child_fit / (child_fit + np.sum(fit_result) - np.min(fit_result)), 5)
    return [child_eval, child_fit, child_prob]


def discard_individual(dicts, key):
    del dicts[key]
    return dicts


def insert_child(dicts, key, values):
    dicts[key] = values
    return dicts


def multiprocess(pool):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x_test, y_test) for p in population))
    return evaluation_result


x = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(0,))

y_test = y[:40000]
x_test = x[:40000]

y_train = y[40000:50000]
x_train = x[40000:50000]

generations = 100  # Number of generation algorithm run.
pop_size = 100
neurons = [4, 7, 10, 1]
number_of_mutate_node = 1
# pool = mp.Pool(mp.cpu_count())
# print(mp.cpu_count())
# fit_result = 0
# while np.mean(fit_result) < 2.5:
#     population = define_model(population_size=pop_size, neurons=neurons)
#     sample_weights = population[0].get_weights()
#     for p in population:
#         p.set_weights(weights_init(sample_weights))
#     # eval_result = evaluation(population, x_test, y_test)
#     eval_result = multiprocess(pool)
#     fit_result = multiprocess_fitness(eval_result)
#     print(np.mean(fit_result))

population = define_model(population_size=pop_size, neurons=neurons)
sample_weights = population[0].get_weights()
for p in population:
    p.set_weights(weights_init(sample_weights))
eval_result = evaluation(population, x_test, y_test)
fit_result = eval_result
prob_pop = selection(fit_result)
dicts = {population[i]: [eval_result[i], fit_result[i], prob_pop[i]] for i in range(0, len(population))}

for p in population:
    if dicts[p][1] == np.max(fit_result):
        best_GA = copy.deepcopy(p)

for g in range(0, generations):
    print('The generation number is: ', g + 1)
    child = breeding(list(dicts.keys()), neurons, dicts, number_of_mutate_node)
    child_values = child_val(child, x_test, y_test, dicts, list(dicts.keys()))

    # Discarding
    for p in list(dicts.keys()):
        if dicts[p][1] == np.min(fit_result):
            discard_individual(dicts, p)
    print(child_values)
    dicts = insert_child(dicts, child, child_values)
    # Updating Probabilities in Dictionary
    for p in list(dicts.keys()):
        dicts[p][2] = round(dicts[p][1] / np.sum([dicts[p][1] for p in list(dicts.keys())]), 5)

best_GA.fit(x_train, y_train, epochs=1, validation_split=0.5)