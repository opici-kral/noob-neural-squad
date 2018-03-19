import time
import copy
import numpy as np

np.random.seed(0)


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation
int2binary = {}
binary_dim = 3
aim_path_1 = np.array([1, 0, 1])
aim_path_2 = np.array([1, 1, 0])
aim_path_3 = np.array([0, 1, 1])
_y_1 = np.array([1])
_y_2 = np.array([2])
_y_3 = np.array([3])


def variables_generator():
    train_set = []
    for i in range(0, 1000):
        rand = np.random.random()
        if rand < .333:
            X = aim_path_1
            y = _y_1
        elif .333 < rand < .666:
            X = aim_path_2
            y = _y_2
        else:
            X = aim_path_3
            y = _y_3
        train_set.append([X, y])
    return train_set


S = variables_generator()
#for i in range(0, len(S)):
#    print(S[i])


#time.sleep(22)
largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 3
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)


# training logic
layer_2_deltas = list()
layer_1_values = list()
layer_1_values.append(np.zeros(hidden_dim))

for j in range(0, len(S)):

    a = S[j][0]
    c = S[j][1]
    d = np.zeros_like(c)

    overallError = 0

   # layer_2_deltas = list()
 #   layer_1_values = list()
  #  layer_1_values.append(np.zeros(hidden_dim))

    for position in range(1):
        # generate input and output
        X = a
        y = c.T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))


        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])


        # decode estimate so we can print it out
        d[0] = np.round(layer_2[0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))


    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(1):
        X = a
        layer_1 = layer_1_values[-1]
        prev_layer_1 = layer_1_values[-2]



        # error at output layer
        layer_2_delta = layer_2_deltas[-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)




        synapse_1_update = synapse_1_update + np.atleast_2d(layer_1 * layer_2_delta[0])
        synapse_h_update = synapse_h_update + np.atleast_2d(prev_layer_1).dot(layer_1_delta)
        synapse_0_update = synapse_0_update + [np.multiply(X[0],layer_1_delta),np.multiply(X[1],layer_1_delta),np.multiply(X[2],layer_1_delta)]
        future_layer_1_delta = layer_1_delta

  #      print("synapse_1_update",synapse_1_update)
  #      print("")
        print("synapse_h_update",synapse_h_update)
  #      print("")
  #      print("synapse_0_update", synapse_0_update)
  #      print("")
  #      print("prev_layer_1",prev_layer_1)
  #      time.sleep(100)

    synapse_0 = synapse_0 + synapse_0_update * alpha
    synapse_1 = synapse_1 + synapse_1_update * alpha
    synapse_h = synapse_h + synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
 #   if (j % 10 == 0):
 #       print("Error:" + str(overallError))
 #       print("_Prev:" + str(a))
 #       print("Guess:" + str(d))
 #       print("_True:" + str(c))
 #       out = 0
 #       for index, x in enumerate(reversed(d)):
 #           out += x * pow(2, index)
 #           # print(str(a_int) + " + " + str(b_int) + " = " + str(out))
 #       print("------------")

print("======= test ========")
X = np.array([0, 0, 0, 0, 0, 0, 0, 0])
q = np.zeros_like(d)
print(q)
