import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 20

TRAIN_IMAGE_FILENAME = './data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = './data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = './data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = './data/mnist/t10k-labels.idx1-ubyte'

num_input_layers = 784
num_hidden_layers = 25
num_output_layers = 10

def read_mnist():
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    # print(f'dimensions of train_images: {train_images.shape}')
    # print(f'dimensions of train_labels: {train_labels.shape}')
    # print(f'dimensions of test_images: {test_images.shape}')
    # print(f'dimensions of test_labels: {test_labels.shape}')

    # # Print a training example
    # print(f'Label for first training example {train_labels[0]}')
    # print(f'---beginning of pattern for first training example---')
    # for line in train_images[0]:
    #     for num in line:
    #         if num > 0:
    #             print('*', end = ' ')
    #         else:
    #             print(' ', end = ' ')
    #     print('')
    # print('---end of pattern for first training example---')

    X_train = train_images.reshape(60000, num_input_layers)
    X_test = test_images.reshape(10000, num_input_layers)

    mean = np.mean(X_train)
    stddev = np.std(X_train)

    # normalizing the data
    X_train = (X_train - mean )/ stddev
    X_test = (X_test - mean) / stddev

    y_train = np.zeros((60000, num_output_layers))
    y_test = np.zeros((10000, num_output_layers))

    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    
    for i, y in enumerate(test_labels):
        # print(f'test label for i:{i} y:{y}')
        y_test[i][y] = 1

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = read_mnist()

# we'll use this to randomize the order of evaluation
index_list = list(range(len(X_train)))

def layer_weights(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
        return weights


hidden_layer_weights = layer_weights(num_hidden_layers, num_input_layers)
hidden_layer_outputs = np.zeros(num_hidden_layers)
hidden_layer_error = np.zeros(num_hidden_layers)

output_layer_weights = layer_weights(num_output_layers, num_hidden_layers)
output_layer_outputs = np.zeros(num_output_layers)
output_layer_error = np.zeros(num_output_layers)

chart_x = []
chart_y_train = []
chart_y_test = []

def show_learning(epoch_number, train_accuracy, test_accuracy):
    global chart_x
    global chart_y_train
    global chart_y_test

    print(f'epoch number: {epoch_number}, training accuracy {train_accuracy: 6.4f}, test accuracy: {test_accuracy: 6.4f}')

    # TODO: can this be put into a tuple? Or a dict including a tuple?
    chart_x.append(epoch_number + 1)
    chart_y_train.append(1.0 - train_accuracy)
    chart_y_test.append(1.0 - test_accuracy)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')

    plt.axis([0, len(chart_x), 0.0, 1.0])

    plt.xlabel('training epochs')
    plt.ylabel('error')

    plt.legend()

    plt.show()

def forward_pass(x):
    global hidden_layer_outputs
    global output_layer_outputs

    #activation function for hidden layer
    for i, w in enumerate(hidden_layer_weights):
        # dot product of the weight and the input value
        z = np.dot(w, x)
        # and we're using tanh as our activation function, so we apply that to our dot result
        hidden_layer_outputs[i] = np.tanh(z)

    # bias is always 1.0, then the calculated values from above
    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_outputs))

    # activation function for output layer
    for i, w in enumerate(output_layer_weights):
        # same as above, dot product of weight and input value
        z = np.dot(w, hidden_output_array)
        # now we're taking the logistic sigmoid for the output
        output_layer_outputs[i] = 1.0/ (1.0 + np.exp(-z))

def backward_pass(ground_truth):
    global hidden_layer_error
    global output_layer_error

    # backprop error for each output neuron
    # and craete an array of all output neuron errors
    for i, y in enumerate(output_layer_outputs):
        error_prime = -(ground_truth[i] - y) # derivative of the loss
        derivative = y * (1.0 - y) # and logistic derivative

        output_layer_error[i] = error_prime * derivative
    
    for i, y in enumerate(hidden_layer_outputs):
        # create array weights connecting the output of the hidden neuron
        # to neurons in the output layer
        error_weights = []

        for w in output_layer_weights:
            error_weights.append(w[i+1])
        
        error_weight_array = np.array(error_weights)

        # backprop error for hidden neuron
        derivative = 1.0 - y**2 # tanh derivative
        weighted_error = np.dot(error_weight_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    global output_layer_weights
    global hidden_layer_weights

    for i, error in enumerate(hidden_layer_error):
        hidden_layer_weights[i] -= (x * LEARNING_RATE * error)

    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_outputs))

    for i, error in enumerate(output_layer_error):
        output_layer_weights[i] -= (hidden_output_array * LEARNING_RATE * error)

# training loop
for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0

    for j in index_list:
        x = np.concatenate((np.array([1.0]), X_train[j]))

        forward_pass(x)

        if output_layer_outputs.argmax() == y_train[j].argmax():
            correct_training_results += 1

        backward_pass(y_train[j])

        adjust_weights(x)

    correct_test_results = 0

    for j in range(len(X_test)):
        x = np.concatenate((np.array([1.0]), X_test[j]))

        forward_pass(x)

        if output_layer_outputs.argmax() == y_test[j].argmax():
            correct_test_results += 1

    show_learning(i, correct_training_results/len(X_train), correct_test_results/len(X_test))

plot_learning()