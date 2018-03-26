import copy
import numpy as np
import time

np.random.seed(0)


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def random_sample_generator():
    sample = list()

    for i in range(0, 1000):
        if np.random.random(1) < .333:
            sample.append(.1)
        elif .333 <= np.random.random(1) <= .666:
            sample.append(.2)
        else:
            sample.append(.3)
    return sample


sample = random_sample_generator()
for i in range(len(sample)):
    print(str(i), " ", sample[i])

int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)

pre_X = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 1]])
pre_y = np.array([[.1], [.1], [.2], [.1], [.2], [.1], [.3], [.3], [.1]])

alpha = 0.1
input_dim = 1
hidden_dim = 7
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
        y = np.array([pre_y[binary_dim - position-1]]).T

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        d[binary_dim - position - 1] = round(layer_2[0][0], 1)
        #d[binary_dim - position - 1] = layer_2[0][0]

        layer_1_values.append(copy.deepcopy(layer_1))

        if j > 9990:
            print("f(", X, ",[", position, "])=",y)
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
print("synapse_0:",synapse_0)
print("")
print("synapse_1:",synapse_1)
print("")
print("synapse_h:",synapse_h)

d = np.zeros_like(pre_y)
overallError = 0
layer_2_deltas = list()
layer_1_values = list()
layer_1_values.append(np.zeros(hidden_dim))
for position in range(binary_dim):

        X = np.array([pre_y[binary_dim - position]])
        y = np.array([pre_y[binary_dim - position-1]]).T

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        d[binary_dim - position - 1] = round(layer_2[0][0], 1)
        #d[binary_dim - position - 1] = layer_2[0][0]

        layer_1_values.append(copy.deepcopy(layer_1))

        if j > 9990:
            print("f(", X, ",[", position, "])=",y)
            print("Error:" + str(overallError))
            print("__Pre:" + str(X))
            print("Guess:" + str(d[binary_dim - position - 1]))
            print("_True:" + str(y))
            print("--------------------")

        future_layer_1_delta = np.zeros(hidden_dim)