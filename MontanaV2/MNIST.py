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
        model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
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


def child_val(child, x, y, dicts, population , file):
    fit_result = []
    for p in population:
        fit_result.append(dicts[p][0])
    print('The mean of fitness is:', np.mean(fit_result))
    print('The best of fitness is:', np.max(fit_result))
    file.write('\nThe mean of fitness is:' + str(np.mean(fit_result)))
    file.write('\nThe best of fitness is:' + str(np.max(fit_result)))
    child_fit = round(child.evaluate(x, y, verbose=0)[1], 5)
    child_prob = round(child_fit / (child_fit + np.sum(fit_result) - np.min(fit_result)), 5)
    return [child_fit, child_prob]


def discard_individual(dicts, key):
    del dicts[key]
    return dicts


def insert_child(dicts, key, values):
    dicts[key] = values
    return dicts


def multiprocess(pool, population, x, y):
    evaluation_result = pool.starmap(Sequential.evaluate, ((p, x, y) for p in population))
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

generations = 100  # Number of generation algorithm run.
pop_size = 50
neurons = [784, 512, 512, 10]
number_of_mutate_node = 180
number_of_child_generate = 2
file = open('report.txt', 'w')
print('The initialization started!')
population = define_model(population_size=pop_size, neurons=neurons)
print('The initialization is DONE!')
sample_weights = population[0].get_weights()

# fit_result = evaluation(population, x_test, y_test)
print('The evaluation started!')
fit_result = multiprocess(pool, population, x_train, y_train)
print('The evaluation is DONE!')
prob_pop = probability(fit_result)
print('Creating the dictionary.')
dicts = {population[i]: [fit_result[i], prob_pop[i]] for i in range(0, len(population))}
print('Dictionary Created!')

# for p in population:
#     if dicts[p][0] == np.min(fit_result):
#         worst = copy.deepcopy(p)

for g in range(0, generations):
    print('The generation number is: ', g + 1)
    file.write('\nThe generation number is:'+str(g + 1))
    children = []
    for c in range(0, number_of_child_generate):
        children.append(breeding(list(dicts.keys()), neurons, dicts, number_of_mutate_node, sample_weights))
    for c in range(0, number_of_child_generate):
        # Discarding
        for p in list(dicts.keys()):
            if dicts[p][0] == np.min(fit_result):
                fit_result.remove(np.min(fit_result))
                dicts = discard_individual(dicts, p)
                break
    for child in children:
        child_values = child_val(child, x_train, y_train, dicts, list(dicts.keys()), file)
        fit_result.append(child_values[0])
        print(child_values)
        file.write('\nThe child values is:'+str(child_values))
        dicts = insert_child(dicts, child, child_values)
    # Updating Probabilities in Dictionary
    for p in list(dicts.keys()):
        dicts[p][1] = round(dicts[p][0] / np.sum([dicts[p][0] for p in list(dicts.keys())]), 5)

# worst.fit(x_train, y_train, epochs=1, validation_split=0.2)

