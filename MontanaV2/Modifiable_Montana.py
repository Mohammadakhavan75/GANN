#for run on cpu
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    a = [weights[z].shape for z in range(0, len(weights))]
    for i in range(0, len(population) - 1, 2):
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
        del m, m1, m2, a
    return model


# Double Checked
def mutate_nodes(population, neurons, number_of_mutation):
    model = define_single_model(neurons)
    m1 = population.get_weights()
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
            # This is working
            m[layers_number][k][neurons_number] = m1[layers_number][k][neurons_number] + random_number
    model.set_weights(m)
    del m1, m
    return model


def evaluation(population, x, y):
    # result of evaluation weights network
    evaluation_result = []
    for p in population:
        evaluation_result.append(round(p.evaluate(x, y, verbose=0, batch_size=60000)[1], 5))
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


def child_val(child, x, y, dicts, population):
    fit_result = []
    for p in population:
        fit_result.append(dicts[p][0])
    child_fit = round(child.evaluate(x, y, verbose=0, batch_size=60000)[1], 5)
    child_prob = round(child_fit / (child_fit + np.sum(fit_result) - np.min(fit_result)), 5)
    return [child_fit, child_prob]


def discard_individual(dicts, key):
    del dicts[key]
    return dicts


def insert_child(dicts, key, values):
    dicts[key] = values
    return dicts


def multiprocess(pool, population, x, y):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x, y , 60000, 0) for p in population))
    return [e[1] for e in evaluation_result]


pool = mp.Pool(4)

x = genfromtxt("E:\\Work\\GANN\\Data\\mydata4.csv", delimiter=',', skip_header=1, usecols=(2, 3, 4, 5))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata4.csv", delimiter=',', skip_header=1, usecols=(1,))

x_train = x[:60000]
y_train = y[:60000]

y_test = y[60000:70000]
x_test = x[60000:70000]

generations = 100  # Number of generation algorithm run.
pop_size = 50
neurons = [4, 7, 10, 1]
number_of_mutate_node = 2

file = open('Montana_Result.txt', 'w')
gen_fit_file = open('gen_fit.txt', 'w')
gen_child_file = open('gen_child.txt', 'w')
gen_mean_file = open('gen_mean.txt', 'w')
gen_max_file = open('gen_max.txt', 'w')
timefile = open('time.txt', 'w')

rs = time.time()
iteration = 0
while iteration < 10:
    print('The initialization started!')
    population = define_model(population_size=pop_size, neurons=neurons)
    print('The initialization is DONE!')
    sample_weights = population[0].get_weights()

    print('The evaluation started!')
    fit_result = multiprocess(pool, population, x_train, y_train)
    print('The evaluation is DONE!')
    prob_pop = probability(fit_result)
    print('Creating the dictionary.')
    dicts = {population[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(population))}
    print('Dictionary Created!')

    for p in population:
        if dicts[p][0] == np.min(fit_result):
            worst = copy.deepcopy(p)
    converge_time = []
    run_time = []
    g = 0
    # for g in range(0, generations):
    while True:
        pop_fit = []
        print('The generation number is: ', g + 1)
        file.write('\nThe generation number is:' + str(g + 1))
        cs = time.time()
        child = breeding(list(dicts.keys()), neurons, dicts, number_of_mutate_node, sample_weights)
        child_values = child_val(child, x_train, y_train, dicts, list(dicts.keys()))
        ce = time.time()
        converge_time.append(ce-cs)
        # Discarding
        for p in list(dicts.keys()):
            if dicts[p][0] == np.min(fit_result):
                dicts = discard_individual(dicts, p)
                fit_result.remove(np.min(fit_result))
                break
        print(child_values)
        file.write('\nThe child values is:' + str(child_values))
        fit_result.append(child_values[0])
        dicts = insert_child(dicts, child, child_values)
        # gen_child.append(dicts[child][0])
        gen_child_file.write('\n' + str(dicts[child][0]))
        # Updating Probabilities in Dictionary
        for p in list(dicts.keys()):
            dicts[p][1] = round(dicts[p][0] / np.sum([dicts[p][0] for p in list(dicts.keys())]), 5)
            pop_fit.append(dicts[p][0])
        print('The mean of fitness is:', np.mean(pop_fit))
        print('The best of fitness is:', np.max(pop_fit))
        file.write('\nThe mean of fitness is:' + str(np.mean(pop_fit)))
        file.write('\nThe best of fitness is:' + str(np.max(pop_fit)))
        file.flush()
        gen_fit_file.write('\n' + str(pop_fit))
        gen_fit_file.flush()
        gen_mean_file.write('\n' + str(np.mean(pop_fit)))
        gen_mean_file.flush()
        gen_max_file.write('\n' + str(np.max(pop_fit)))
        gen_max_file.flush()
        g += 1
        if np.max(pop_fit) > 0.8:
            break

    re = time.time()
    run_time.append(re-rs)

    file.write('\nStart of SGD')
    rs = time.time()
    while True:
        call = worst.fit(x_train, y_train, batch_size=60000)
        file.write(str(call.history['acc'][0])+'\n')
        file.flush()
        if call.history['acc'][0] > np.max(pop_fit):
            break
    iteration += 1
    re = time.time()
    run_time.append(re-rs)
    timefile.write('The run time of SGD is:'+str(run_time[1]))
    timefile.write('\nThe run time of GA is:' + str(run_time[0]))
    timefile.write('\nThe convergance time of GA is:' + str(np.sum(converge_time)))
    timefile.write('\n**************\n****  '+str(iteration)+'  ****\n**************\n')
    timefile.flush()
