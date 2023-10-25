import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 20

# Where dez images
TRAIN_IMAGE_FILENAME = "data/mnist/train-images.idx3-ubyte"
TRAIN_LABEL_FILENAME = "data/mnist/train-labels.idx1-ubyte"
TEST_IMAGE_FILENAME = "data/mnist/t10k-images.idx3-ubyte"
TEST_LABEL_FILENAME = "data/mnist/t10k-labels.idx1-ubyte"


def read_mnist():
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    # flattens image to 1d instead of 2d
    x_train = train_images.reshape(60000, 784)
    x_test = test_images.reshape(10000, 784)

    mean = np.mean(x_train)
    stddev = np.std(x_train)

    # standardize values. Remember to standardize both sets around your x_train mean/stddev!
    # if you standardize each set around its own mean/stddev the model won't be able to infer well
    # in general, any transform you do to your training set, you MUST do to our test/PROD sets
    # this in particular gets us around 0
    # subtracting the mean, makes the mean 0 (if the mean is 1, and you subtract 1, you remove the mean)
    # dividing by the stddev, then scrunches the range in by that amount
    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev

    # one-hot
    # initialize everything to zero.  WE have 10 possible 1s for each set, since we're guessing
    # numbers 1-10
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))

    # this looks at the training labels, which are index-value, and sets
    # one of the 10 possible 1s in y_train to 1
    # so if the label for index 3 is 6, it would set y_train[3][6] = 1, or 0000001000
    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read_mnist()
index_list = list(range(len(x_train)))


# Randomly initialize neuron weights from -1.0 to 1.0
# This also leaves the bias weight (at pos 0), as a zero
# We want to randomly intitialize to break up the symmetry. If the
# weights are all the same number, they would all get the same adjustment during backprop.
def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))

    for i in range(neuron_count):
        for j in range(1, (input_count + 1)):
            weights[i][j] = np.random.uniform(-1.0, 1.0)

    return weights


# 25 neurons, 784 inputs for our hidden layer
hidden_layer_w = layer_w(25, 784)
# 25 neurons, 25 outputs
hidden_layer_y = np.zeros(25)
# also, 25 neurons, 25 error values
hidden_layer_error = np.zeros(25)

# the hidden layer has 25 outputs, so the output layer has 25 inputs
# it has 10 outputs, since we're determining values 0-9
output_layer_w = layer_w(10, 25)
output_layer_y = np.zeros(10)
output_layer_error = np.zeros(10)

# here we're going to show our learning
chart_x = []
chart_y_train = []
chart_y_test = []


# text representation of learning during one epoch
# we store the data in global vars to use later
def show_learning(epoch_no, train_acc, test_acc):
    global chart_x
    global chart_y_train
    global chart_y_test

    print(
        "epoch no: ",
        epoch_no,
        ", train_acc: ",
        "%6.4f" % train_acc,
        ", test_acc: ",
        "%6.4f" % test_acc,
    )

    chart_x.append(epoch_no + 1)
    chart_y_train.append(1.0 - train_acc)
    chart_y_test.append(1.0 - test_acc)


def plot_learning():
    plt.plot(chart_x, chart_y_train, "r-", label="training error")
    plt.plot(chart_x, chart_y_test, "b-", label="test error")

    plt.axis([0, len(chart_x), 0, 1.0])

    plt.xlabel("training epochs")
    plt.ylabel("error")

    plt.legend()

    plt.show()


def forward_pass(x):
    global hidden_layer_y
    global output_layer_y

    # Let each neuron in the hidden layer look at every image
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)

    # add in our bias term!
    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

    # show the output of each hidden layer, to each input of the output layer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))


def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error

    # backproping error
    # starting backwards, compute derivative of loss function for each output neuron
    for i, y in enumerate(output_layer_y):
        error_prime = -(y_truth[i] - y)  # loss derivative
        derivative = y * (1.0 - y)  # logistic derivative

        output_layer_error[i] = error_prime * derivative

    # output_layer_w is a 2d array, 10 rows (one for each neuron), 25 columns (one for each output of hidden y)
    # hidden_layer_Y is a 1d array with 25 values
    for i, y in enumerate(hidden_layer_y):
        error_weights = []

        # here we are adding each row of the output layer weights to the error_weights var

        for w in output_layer_w:
            error_weights.append(w[i + 1])

        error_weight_array = np.array(error_weights)

        # time to backprop the error
        derivative = 1.0 - y**2  # tanh derivative
        weighted_error = np.dot(error_weight_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w

    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LEARNING_RATE * error) # updating our weights
    
    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error) 

# training loop
for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0

    # don't really like that we're using global vars 
    # below doesn't give a good feel as to what's going on for a programmer
    for j in index_list:
        # add bias
        x = np.concatenate((np.array([1.0]), x_train[j]))

        forward_pass(x)

        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training_results += 1
        
        backward_pass(y_train[j])

        adjust_weights(x)

    correct_test_results = 0
    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))

        forward_pass(x)

        if output_layer_y.argmax() == y_test[j].argmax():
            correct_test_results += 1

    show_learning(i, correct_training_results/len(x_train),
                  correct_test_results/len(x_test))
    
plot_learning()