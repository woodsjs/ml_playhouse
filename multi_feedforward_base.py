"""
Based on code from Learning Deep Learning

This is a multineuron feedforward network
"""

import numpy as np

np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0,1,2,3]

X_train = [np.array([1.0, -1.0, -1.0]),
            np.array([1.0, -1.0, 1.0]),
            np.array([1.0, 1.0, -1.0]),
            np.array([1.0, 1.0, 1.0])]
y_train = [0.0, 1.0, 1.0, 0.0]


# initializing a neuron passed in with random weights. 
# always start with random weights! If we don't, the output of the neurons is the 
# same, the adjustments would be the same, and it would be as if we only had a 
# single neuron. This forces each neuron to work independently
# Note we only initialize the 2nd and 3rd weight to random, and 
# leave the first weight at 0.0.  The first weight is the BIAS. It's OK for this to
# be zero, as the other weights will sufficiently randomize each neuron.
def init_neuron_weights(weight_count):
    weights = np.zeros(weight_count + 1)
    for i in range(1, weight_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights

# the weight count is without the bias. So we have 2 weights, but it will have 3 numbers
neuron_weights = [init_neuron_weights(2), init_neuron_weights(2), init_neuron_weights(2)]
neuron_outputs = [0,0,0]
neuron_errors = [0,0,0]

def show_learning():
    print('Current Weights')
    for i, w in enumerate(neuron_weights):
        print(
            f'neuron {i} : w0={w[0]: 5.2f}, w1={w[1]: 5.2f }, w2={w[2]: 5.2f}')
        print(f'{"-"*10}')

def forward_pass(x):
    global neuron_outputs

    # work on layer one
    neuron_outputs[0] = np.tanh(np.dot(neuron_weights[0], x)) 
    neuron_outputs[1] = np.tanh(np.dot(neuron_weights[1], x))

    # output for layer two, or the output layer
    layer_two_inputs = np.array([1.0, neuron_outputs[0], neuron_outputs[1]])

    # apply the current weights to the input using a dot product
    layer_two_z = np.dot(neuron_weights[2], layer_two_inputs)

    # for our sigmoid output, we calculate the exponential of the layer two input
    # we actually invert it, so that if out input is negative and we need to adjust
    # positive, it works
    neuron_outputs[2] = 1.0/(1.0 + np.exp(-layer_two_z))

# We're taking our outputs and using them to work backwards to derive
# our weight adjustments
# SO, the output of the network's output neuron derivative and error determine
# how much to adjust the neurons FEEDING that output neuron to minimize the OUTPUT NEURON's
# output. 
def backward_pass(y_truth):
    global neuron_errors

    # derivative of loss function for our output neuron
    # or expected_y minus y_hat
    error_prime = -(y_truth - neuron_outputs[2])
    # logistic derivative of our output neuron's activation function
    derivative = neuron_outputs[2] * (1.0 - neuron_outputs[2])
    neuron_errors[2] = error_prime * derivative

    # hidden layer
    #tanh derivative of our first neuron's activation function
    derivative = 1.0 - neuron_outputs[0]**2
    # we're just figuring out what our weight adjustments will be here
    # so take the first weight for the output neuron, which will be the weight
    # applied to the output of the first input neurons output going into the 
    # output neuron (you read that right), multiply that by the output neuron
    # error and derivative.
    neuron_errors[0] = neuron_weights[2][1] * neuron_errors[2] * derivative

    #Second verse, same as the first
    derivative = 1.0 - neuron_outputs[1]**2
    neuron_errors[1] = neuron_weights[2][2] * neuron_errors[2] * derivative

def adjust_weights(x):
    global neuron_weights

    # seems legit
    neuron_weights[0] -= (x * LEARNING_RATE * neuron_errors[0])
    neuron_weights[1] -= (x * LEARNING_RATE * neuron_errors[1])

    # we have to use ALL inputs into a neuron when adjusting the weight!!!!
    layer_two_inputs = np.array([1.0, neuron_outputs[0], neuron_outputs[1]])
    neuron_weights[2] -= (layer_two_inputs * LEARNING_RATE * neuron_errors[2])
