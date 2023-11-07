// Once again, we're writing cpp based on python.
// What you looking at?
#include "MnistDataloader.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>
#include <cmath>

// np.random.seed(7)
float LEARNING_RATE = 0.01;
int EPOCHS = 20;

// # Where dez images
std::string TRAIN_IMAGE_FILENAME = "../data/mnist/train-images.idx3-ubyte";
std::string TRAIN_LABEL_FILENAME = "../data/mnist/train-labels.idx1-ubyte";
std::string TEST_IMAGE_FILENAME = "../data/mnist/t10k-images.idx3-ubyte";
std::string TEST_LABEL_FILENAME = "../data/mnist/t10k-labels.idx1-ubyte";

void read_mnist(std::string input_image_filename, std::string input_label_filename, std::vector<uint16_t> &labels, std::vector<std::vector<std::vector<uint8_t>>> &images)
{
    MnistDataloader mnistDL;

    mnistDL.read_images_labels(input_image_filename, input_label_filename, labels, images);
}

// # Randomly initialize neuron weights from -1.0 to 1.0
// # This also leaves the bias weight (at pos 0), as a zero
// # We want to randomly intitialize to break up the symmetry. If the
// # weights are all the same number, they would all get the same adjustment during backprop.
std::vector<std::vector<double>> init_neuron_weights(int neuron_count, int input_count)
{
    // we increase the input count by one, so we can have the first value our bias of zero
    std::vector<std::vector<double>> weights(neuron_count, std::vector<double>(input_count + 1));
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> distDoub(-1.0, 1.0); // distribution in range [-1.0, 1.0];

    for (auto i = 0; i < neuron_count; i++)
    {
        for (auto j = 1; j < input_count; j++ )
            weights[i][j] = distDoub(rng); // needs to be some rando thing
    }

    return weights;
}

// # 25 neurons, 784 inputs for our hidden layer
// hidden_layer_w = layer_w(25, 784)
// # 25 neurons, 25 outputs
// hidden_layer_y = np.zeros(25)
// # also, 25 neurons, 25 error values
// hidden_layer_error = np.zeros(25)

// # the hidden layer has 25 outputs, so the output layer has 25 inputs
// # it has 10 outputs, since we're determining values 0-9
// output_layer_w = layer_w(10, 25)
// output_layer_y = np.zeros(10)
// output_layer_error = np.zeros(10)

// # here we're going to show our learning
// chart_x = []
// chart_y_train = []
// chart_y_test = []

// # text representation of learning during one epoch
// # we store the data in global vars to use later
// def show_learning(epoch_no, train_acc, test_acc):
//     global chart_x
//     global chart_y_train
//     global chart_y_test

//     print(
//         "epoch no: ",
//         epoch_no,
//         ", train_acc: ",
//         "%6.4f" % train_acc,
//         ", test_acc: ",
//         "%6.4f" % test_acc,
//     )

//     chart_x.append(epoch_no + 1)
//     chart_y_train.append(1.0 - train_acc)
//     chart_y_test.append(1.0 - test_acc)

// def plot_learning():
//     plt.plot(chart_x, chart_y_train, "r-", label="training error")
//     plt.plot(chart_x, chart_y_test, "b-", label="test error")

//     plt.axis([0, len(chart_x), 0, 1.0])

//     plt.xlabel("training epochs")
//     plt.ylabel("error")

//     plt.legend()

//     plt.show()

// def forward_pass(x):
//     global hidden_layer_y
//     global output_layer_y

//     # Let each neuron in the hidden layer look at every image
//     for i, w in enumerate(hidden_layer_w):
//         z = np.dot(w, x)
//         hidden_layer_y[i] = np.tanh(z)

//     # add in our bias term!
//     hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

//     # show the output of each hidden layer, to each input of the output layer
//     for i, w in enumerate(output_layer_w):
//         z = np.dot(w, hidden_output_array)
//         output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

// def backward_pass(y_truth):
//     global hidden_layer_error
//     global output_layer_error

//     # backproping error
//     # starting backwards, compute derivative of loss function for each output neuron
//     for i, y in enumerate(output_layer_y):
//         error_prime = -(y_truth[i] - y)  # loss derivative
//         derivative = y * (1.0 - y)  # logistic derivative

//         output_layer_error[i] = error_prime * derivative

//     # output_layer_w is a 2d array, 10 rows (one for each neuron), 25 columns (one for each output of hidden y)
//     # hidden_layer_Y is a 1d array with 25 values
//     for i, y in enumerate(hidden_layer_y):
//         error_weights = []

//         # here we are adding each row of the output layer weights to the error_weights var

//         for w in output_layer_w:
//             error_weights.append(w[i + 1])

//         error_weight_array = np.array(error_weights)

//         # time to backprop the error
//         derivative = 1.0 - y**2  # tanh derivative
//         weighted_error = np.dot(error_weight_array, output_layer_error)
//         hidden_layer_error[i] = weighted_error * derivative

// def adjust_weights(x):
//     global output_layer_w
//     global hidden_layer_w

//     for i, error in enumerate(hidden_layer_error):
//         hidden_layer_w[i] -= (x * LEARNING_RATE * error) # updating our weights

//     hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

//     for i, error in enumerate(output_layer_error):
//         output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error)

// # training loop
// for i in range(EPOCHS):
//     np.random.shuffle(index_list)
//     correct_training_results = 0

//     # don't really like that we're using global vars
//     # below doesn't give a good feel as to what's going on for a programmer
//     for j in index_list:
//         # add bias
//         x = np.concatenate((np.array([1.0]), x_train[j]))

//         forward_pass(x)

//         if output_layer_y.argmax() == y_train[j].argmax():
//             correct_training_results += 1

//         backward_pass(y_train[j])

//         adjust_weights(x)

//     correct_test_results = 0
//     for j in range(len(x_test)):
//         x = np.concatenate((np.array([1.0]), x_test[j]))

//         forward_pass(x)

//         if output_layer_y.argmax() == y_test[j].argmax():
//             correct_test_results += 1

//     show_learning(i, correct_training_results/len(x_train),
//                   correct_test_results/len(x_test))

// plot_learning()

int main(void)
{

    // our mnist reader will push out our labels (y_train, y_test), as well as our images (x_train, x_test)
    std::vector<uint16_t> y_train;
    std::vector<std::vector<std::vector<uint8_t>>> x_train;

    std::vector<uint16_t> y_test;
    std::vector<std::vector<std::vector<uint8_t>>> x_test;

    read_mnist(TRAIN_IMAGE_FILENAME, TRAIN_LABEL_FILENAME, y_train, x_train);
    read_mnist(TEST_IMAGE_FILENAME, TEST_LABEL_FILENAME, y_test, x_test);

    // index_list = list(range(len(x_train)))

    // get our weights
    std::vector<std::vector<double>> neuron_weights;
    neuron_weights = init_neuron_weights(3, 24);

    return 0;
}