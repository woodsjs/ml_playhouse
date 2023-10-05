import numpy as np

np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0,1,2,3]

x_train = [np.array(1.0, -1.0, -1.0),
           np.array(1.0, -1.0, 1.0),
           np.array(1.0, 1.0, -1.0),
           np.array(1.0, 1.0, 1.0)]
y_train = [0.0, 1.0, 1.0, 0.0]

# Randomly initialize neuron weights from -1.0 to 1.0
# This also leaves the bias weight (at pos 0), as a zero
# We want to randomly intitialize to break up the symmetry. If the
# weights are all the same number, they would all get the same adjustment during backprop.
def init_neuron_weights(input_count):
    weights = np.zeros(input_count+1)

    for i in range(1, (input_count+1)):
        weights[i] = np.random.uniform(-1.0, 1.0)

    return weights

neuron_weights = [init_neuron_weights(2), init_neuron_weights(2), init_neuron_weights(2)]
neuron_y = [0,0,0]
neuron_error = [0,0,0]

def show_learning(w):
    global color_index

    for i, w in enumerate(n_w):
        print('neuron ', i, 'w0 = ', '%5.2f' % w[0], ', w1 = ' , '%5.2f' %w[1], ', w2 = ', '%5.2f' %w[2])
    
    print("-"*14)

    # if color_index == 0:
    #     plt.plot([1.0], [1.0], 'b_', markersize=12)
    #     plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
    #     plt.axis([-2, 2, -2, 2])
    #     plt.xlabel('x1')
    #     plt.ylabel('x2')

    # x = [-2.0, 2.0]

    # if abs(w[2]) < 1e-5:
    #     y = [-w[1]/(1e-5)*(-2.0)+(-w[0]/(1e-5)),
    #          -w[1]/(1e-5)*((2.0)+(-w[0]/(1e-5)))]
    # else:
    #     y = [-w[1]/w[2]*(x[0])+(-w[0]/w[2]),
    #          -w[1]/w[2]*(x[1])+(-w[0]/w[2])]
        
    # plt.plot(x, y, color_list[color_index])

    # if color_index < (len(color_list) - 1):
    #     color_index += 1

# At the moment this is a two layer, three neuron network
# Layer one, the hidden layer, is two neurons.  Layer two is a single neuron output layer
def forward_pass(x):
    global neuron_y

    # first layer result is the tanh of our weights and input values
    neuron_y[0] = np.tanh(np.dot(neuron_weights[0], x))
    neuron_y[1] = np.tanh(np.dot(neuron_weights[1], x))

    # this is really just to make it clear what our second neuron input looks like
    neuron2_inputs = np.array([1.0, neuron_y[0], neuron_y[1]])

    # This doesn't feel good, but OK let's just brute our second layer at the same time
    # optimization would be to specify the number of layers and neurons per layer, like a good 
    # lib does for us, but the goal here IS to see it clearly.  Sometimes that means coding
    # it long form, then refactor
    z2 = np.dot(neuron_weights[2], neuron2_inputs)

    neuron_y[2] = 1.0/(1.0+np.exp(-z2))