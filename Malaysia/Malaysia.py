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
from keras.datasets import mnist
import keras
from collections import OrderedDict


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
    nm1 = [np.empty(a[z]) for z in range(0, len(a))]
    nm2 = [np.empty(a[z]) for z in range(0, len(a))]
    for i in range(0, len(population) - 1, 2):
        models = define_model(2, neurons)
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
        del models, m1, m2
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
    return population


# Checked
def selecting_for_pool(population, dicts):
    parent = []
    while len(parent) < len(population):
        for p in reversed(population):
            if dicts[p][1] > np.random.uniform(0, 1) and len(parent) < len(population):
                parent.append(p)
    return parent


# Checked (Each paired parents are different)
def mating_pool(population, neurons, dicts, weights):
    paired_parents = []
    parents = selecting_for_pool(population, dicts)
    while len(paired_parents) < len(parents):
        t = np.random.choice(parents, 2, False)
        paired_parents.append(t[0])
        paired_parents.append(t[1])
    new_pop = crossover_nodes_make2child(paired_parents, neurons, weights)
    return new_pop


# Checked
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
        evaluation_result.append(round(p.evaluate(x, y, verbose=0, batch_size=60000)[1], 5))
    return evaluation_result


# Checked
def probability(fitness):
    return [round(x / np.sum(fitness), 5) for x in fitness]


# Checked and Order is correct
def multiprocess(pool, population, x, y):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x, y, 60000, 0) for p in population))
    return [e[1] for e in evaluation_result]


pool = mp.Pool(4)

x = genfromtxt("E:\\Work\\GANN\\Data\\mydata4.csv", delimiter=',', skip_header=1, usecols=(2, 3, 4, 5))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata4.csv", delimiter=',', skip_header=1, usecols=(1,))

y_train = y[:60000]
x_train = x[:60000]

y_test = y[60000:70000]
x_test = x[60000:70000]

generations = 100  # Number of generation algorithm run.
pop_size = 50
neurons = [4, 7, 10, 1]
number_of_mutate_node = 2

print('The initialization started!')
population = define_model(population_size=pop_size, neurons=neurons)
sample_weights = population[0].get_weights()
print('The initialization is DONE!')

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

gen_fit = []
gen_child = []
gen_mean = []
gen_max = []
file = open('report.txt', 'w')
gen_fit_file = open('gen_fit.txt', 'w')
gen_child_file = open('gen_child.txt', 'w')
gen_mean_file = open('gen_mean.txt', 'w')
gen_max_file = open('gen_max.txt', 'w')
converge_time = []
run_time = []
rs = time.time()
g = 0
# for g in range(0, generations):
while True:
    print('The generation number is: ', g + 1)
    file.write('\nThe generation number is:' + str(g + 1))
    dicts = OrderedDict(sorted(dicts.items(), key=lambda x: x[1]))
    cs = time.time()
    new_pop = mating_pool(list(dicts.keys()), neurons, dicts, sample_weights)
    final_pop = mutate_pool(new_pop, neurons, number_of_mutate_node)
    ce = time.time()
    converge_time.append(ce - cs)
    dicts.clear()

    fit_result = multiprocess(pool, final_pop, x_train, y_train)
    prob_pop = probability(fit_result)
    dicts = {final_pop[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(final_pop))}
    print('The mean of fitness is:', np.mean(fit_result))
    print('The best of fitness is:', np.max(fit_result))
    file.write('\nThe mean of fitness is:' + str(np.mean(fit_result)))
    file.write('\nThe best of fitness is:' + str(np.max(fit_result)))
    file.flush()
    gen_fit_file.write('\n' + str(fit_result))
    gen_fit_file.flush()
    gen_mean_file.write('\n' + str(np.mean(fit_result)))
    gen_mean_file.flush()
    gen_max_file.write('\n' + str(np.max(fit_result)))
    gen_max_file.flush()
    g += 1
    # gen_fit.append(fit_result)
    # gen_mean.append(np.mean(fit_result))
    # gen_max.append(np.max(fit_result))
    if np.max(fit_result) > 0.8:
        break

re = time.time()
run_time.append(re-rs)

# for item in gen_child:
#     gen_child_file.write('\n'+str(item))
# for item in gen_fit:
#     gen_fit_file.write('\n'+str(item))
# for item in gen_mean:
#     gen_mean_file.write('\n'+str(item))
# for item in gen_max:
#     gen_max_file.write('\n'+str(item))

file.write('\nStart of SGD')
rs = time.time()
while True:
    call = worst.fit(x_train, y_train, batch_size=60000)
    file.write(str(call.history['acc'][0])+'\n')
    if call.history['acc'][0] > np.max(fit_result):
        break

re = time.time()
run_time.append(re-rs)
with open('time.txt', 'w') as timefile:
    timefile.write('The run time of SGD is:'+str(run_time[1]))
    timefile.write('\nThe run time of GA is:' + str(run_time[0]))
    timefile.write('\nThe convergance time of GA is:' + str(np.sum(converge_time)))
