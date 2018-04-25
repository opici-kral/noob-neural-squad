import copy
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from itertools import product

np.random.seed(466899297)

int2binary = {}
binary_dim = 3


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# generate random vector-path set
def random_sample_generator():
    sample_set = list()
    one_sub_sample = list()
    for j in range(0, 1001):
        for i in range(0, binary_dim + 1):
            if np.random.random(1) < .333:
                one_sub_sample.append([.1])
            elif .333 <= np.random.random(1) <= .666:
                one_sub_sample.append([.2])
            else:
                one_sub_sample.append([.3])
        sample_set.append(one_sub_sample)
        one_sub_sample = list()
    return sample_set


def variations_generator(x):
    sample = list()
    for i in product(x, repeat=len(x)+1):
        sample.append(i)

    return sample


basic_set = [[.0], [.1], [.9]]

basic_sample = variations_generator(basic_set)

for i in range(0, 80):
    print(basic_sample[i])

sample = random_sample_generator()


# for i in range(len(sample)):
#    print(str(i), " ", sample[i])


def transcript_to_coordinates():
    # text_file = open("my_maze.txt", "w+")
    sample_set = list()
    for j in range(len(sample)):
        x = 0
        y = 0
        path = list()
        for i in range(len(sample[j])):
            if sample[j][i][0] == .1:
                y += 1
            elif sample[j][i][0] == .2:
                x += 1
            else:
                x += -1
            path.append([x, y])
        sample_set.append(path)
        # print(path)
        # text_file.write("%s\n" % path)
    # text_file.close()
    return sample_set


my_sample = transcript_to_coordinates()


# for i in range(len(my_sample)):
#    print(my_sample[i])


def cre_sample_for_graph(my_sample):
    sampler = list()
    for j in range(len(my_sample)):
        x1 = list()
        y1 = list()
        for i in range(len(my_sample[j])):
            x1.append(my_sample[j][i][0])
            y1.append(my_sample[j][i][1])
        sampler.append([x1, y1])
    return sampler


path_set = cre_sample_for_graph(my_sample)

print("")
# print(path_set)

for i in range(len(path_set)):
    plt.scatter(path_set[i][0], path_set[i][1])
    plt.plot(path_set[i][0], path_set[i][1])

# plt.ylabel('Y')
# plt.xlabel('x')
plt.show()

time.sleep(1)

largest_number = pow(2, binary_dim)

pre_y = np.array([[.1], [.0], [.9], [.9]])

alpha = 0.1
input_dim = 1
hidden_dim = 21
output_dim = 1

synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

print("==============tunning==============")
for j in range(10000):

    d = np.zeros_like(pre_y)
    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):

        X = np.array([pre_y[binary_dim - position]])
        y = np.array([pre_y[binary_dim - position - 1]]).T

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        d[binary_dim - position - 1] = round(layer_2[0][0], 1)
        # d[binary_dim - position - 1] = layer_2[0][0]

        layer_1_values.append(copy.deepcopy(layer_1))

        if j > 9990:
            print("f(", X, ",[", position, "])=", y)
            print("Error:" + str(overallError))
            print("__Pre:" + str(X))
            print("Guess:" + str(d[binary_dim - position - 1]))
            print("_True:" + str(y))
            print("--------------------")

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([pre_y[position]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        layer_2_delta = layer_2_deltas[-position - 1]

        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
            synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        synapse_1_update = synapse_1_update + np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update = synapse_h_update + np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update = synapse_0_update + X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 = synapse_0 + synapse_0_update * alpha
    synapse_1 = synapse_1 + synapse_1_update * alpha
    synapse_h = synapse_h + synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

print("==============calibrated==============")
print("synapse_0:", synapse_0)
print("")
print("synapse_1:", synapse_1)
print("")
print("synapse_h:", synapse_h)
print("")
print("sample[0]", sample[0])
print("======================================")

sample_output = list()

for i in range(0, 80):
    d = np.zeros_like(pre_y)
    pre_y = basic_sample[0]
    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    print("-----------/submitting this/------------")
    print("------>", basic_sample[i][0])
    print("------>", basic_sample[i][1])
    print("------>", basic_sample[i][2])
    print("------>", basic_sample[i][3])
    print("----------------------------------------")
    output_sample_one = list()

    for position in range(binary_dim):

        X = np.array([basic_sample[i][binary_dim - position]])
        # print(position, ".")
        y = np.array([basic_sample[i][binary_dim - position - 1]]).T

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        d[binary_dim - position - 1] = round(layer_2[0][0], 1)
        output_sample_one.append(d[binary_dim - position - 1])
        # d[binary_dim - position - 1] = layer_2[0][0]

        layer_1_values.append(copy.deepcopy(layer_1))

        if 1 == 1:
            print("f(", X, ",[", position, "])=", y)
            print("Error:" + str(overallError))
            print("__Pre:" + str(X))
            print("Guess:" + str(d[binary_dim - position - 1]))
            print("_True:" + str(y))

            print("--------------------")

        future_layer_1_delta = np.zeros(hidden_dim)
    print("loooo:", output_sample_one)
    sample_output.append(output_sample_one)

for i in range(0, len(sample_output)):
    print(sample_output[i])
