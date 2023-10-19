// Once again, we're writing cpp based on python.
// What you looking at?

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>
#include <cmath>

// # Randomly initialize neuron weights from -1.0 to 1.0
// # This also leaves the bias weight (at pos 0), as a zero
// # We want to randomly intitialize to break up the symmetry. If the
// # weights are all the same number, they would all get the same adjustment during backprop.
// def init_neuron_weights(input_count):
std::vector<double> init_neuron_weights(int input_count)
{
    std::vector<double> weights(input_count + 1, 0.0);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> distDoub(-1.0, 1.0); // distribution in range [-1.0, 1.0];

    for (auto i = 1; i < weights.size(); i++)
    {
        weights[i] = distDoub(rng); // needs to be some rando thing
    }

    return weights;
}

void show_learning(std::vector<std::vector<double>> weights)
{
    // std::cout << "weights size " << weights.size();
    for (auto i = 0.0; i < weights.size(); i++)
    {
        std::cout.precision(2);
        std::cout << "neuron " << i << " w0 = " << weights[i][0] << " w1= " << weights[i][1] << " w[2] = " << weights[i][2] << std::endl;
    }
}

// # At the moment this is a two layer, three neuron network
// # Layer one, the hidden layer, is two neurons.  Layer two is a single neuron output layer
// def forward_pass(x):
std::vector<double> forward_pass(std::vector<double> x, std::vector<double> neuron_y, std::vector<std::vector<double>> neuron_weights)
{
    // dot product. refactor this ugliness
    std::vector<double> mult_intermediate(neuron_weights[0].size(), 0);
    std::transform(x.begin(), x.end(), neuron_weights[0].begin(), mult_intermediate.begin(), [](auto x, auto y)
                   { return x * y; });
    auto z = std::reduce(mult_intermediate.begin(), mult_intermediate.end(), 0.0, [](auto x, auto y)
                         { return (x + y); });

    //  first layer result is the tanh of our weights and input values
    neuron_y[0] = std::tanh(z);

    // dot product. refactor this ugliness
    std::transform(x.begin(), x.end(), neuron_weights[1].begin(), mult_intermediate.begin(), [](auto x, auto y)
                   { return x * y; });
    z = std::reduce(mult_intermediate.begin(), mult_intermediate.end(), 0.0, [](auto x, auto y)
                    { return (x + y); });

    neuron_y[1] = std::tanh(z);

    //    this is really just to make it clear what our second neuron input looks like
    std::vector<double> neuron2_inputs = {1.0, neuron_y[0], neuron_y[1]};

    //     # This doesn't feel good, but OK let's just brute our second layer at the same time
    //     # optimization would be to specify the number of layers and neurons per layer, like a good
    //     # lib does for us, but the goal here IS to see it clearly.  Sometimes that means coding
    //     # it long form, then refactor
    // dot product. refactor this ugliness
    std::transform(neuron2_inputs.begin(), neuron2_inputs.end(), neuron_weights[2].begin(), mult_intermediate.begin(), [](auto x, auto y)
                   { return x * y; });
    z = std::reduce(mult_intermediate.begin(), mult_intermediate.end(), 0.0, [](auto x, auto y)
                    { return (x + y); });
    // not sure if this is the right way to do this in c++
    neuron_y[2] = 1.0 / (1.0 + std::exp(-z));

    return neuron_y;
}

// # now let's adjust our weights based on the error
// def backward_pass(y_truth):
std::vector<double> backward_pass(double y_truth, std::vector<double> neuron_y, std::vector<double> neuron_error, std::vector<std::vector<double>> neuron_weights)
{

    // your loss function derivative, from the output layer
    double error_prime = -(y_truth - neuron_y[2]);

    // and the logistic derivative. Remember, the derivatives build on each other through the layers
    double derivative = neuron_y[2] * (1.0 - neuron_y[2]);
    neuron_error[2] = error_prime * derivative;

    // tanh derivative, for the hidden layer
    derivative = 1.0 - pow(neuron_y[0], 2);
    neuron_error[0] = neuron_weights[2][1] * neuron_error[2] * derivative;

    derivative = 1.0 - pow(neuron_y[1], 2);
    neuron_error[1] = neuron_weights[2][2] * neuron_error[2] * derivative;

    return neuron_error;
}

std::vector<std::vector<double>> adjust_weights(std::vector<double> neuron_y, std::vector<double> neuron_error, std::vector<std::vector<double>> neuron_weights, float LEARNING_RATE, std::vector<double> &x_train)
{

    // what's the better way?
    std::vector<double> intermediate_results(neuron_weights[0].size(), 0);
    std::transform(x_train.begin(), x_train.end(), neuron_weights[0].begin(), intermediate_results.begin(), [&](auto x, auto y)
                   { return y - (x * LEARNING_RATE * neuron_error[0]); });

    neuron_weights[0] = intermediate_results;

    std::transform(x_train.begin(), x_train.end(), neuron_weights[1].begin(), intermediate_results.begin(), [&](auto x, auto y)
                   { return y - (x * LEARNING_RATE * neuron_error[1]); });

    neuron_weights[1] = intermediate_results;

    std::vector<double> n2_inputs = {1.0, neuron_y[0], neuron_y[1]};
    std::transform(n2_inputs.begin(), n2_inputs.end(), neuron_weights[2].begin(), intermediate_results.begin(), [&](auto x, auto y)
                   { return y - (x * LEARNING_RATE * neuron_error[2]); });

    neuron_weights[2] = intermediate_results;

    return neuron_weights;
}

int main()
{
    // # define our hyperparams

    float LEARNING_RATE = 0.1;
    std::vector<int> index_list = {0, 1, 2, 3}; // will be used to randomize order of x_train

    // training examples.Ultimately, like the hyperparams, these should be passed in.
    // here we include in code as we write it
    // First element in the vector v must be 1, our Bias value
    // Len of w, x is n+1 for n inputs

    std::vector<std::vector<double>> x_train = {{1.0, -1.0, -1.0},
                                                {1.0, -1.0, 1.0},
                                                {1.0, 1.0, -1.0},
                                                {1.0, 1.0, 1.0}};
    std::vector<double> y_train = {0.0, 1.0, 1.0, 0.0}; // ground truth

    std::vector<std::vector<double>> neuron_weights = {init_neuron_weights(2), init_neuron_weights(2), init_neuron_weights(2)};
    std::vector<double> neuron_y(3, 0.0);
    std::vector<double> neuron_error(3, 0.0);
    // print initial weights
    show_learning(neuron_weights);

    // This is the meat and potatoes Perceptron training loop
    bool all_correct = false;

    while (!all_correct)
    {
        all_correct = true;

        // randomize the order of the training data intake
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(index_list.begin(), index_list.end(), g);

        // Here we're learning from our training data
        for (auto i : index_list)
        {
            neuron_y = forward_pass(x_train[i], neuron_y, neuron_weights);
            neuron_error = backward_pass(y_train[i], neuron_y, neuron_error, neuron_weights);
            neuron_weights = adjust_weights(neuron_y, neuron_error, neuron_weights, LEARNING_RATE, x_train[i]);

            show_learning(neuron_weights);
        }

        // let's shuffle again, for good measure
        // std::shuffle(index_list.begin(), index_list.end(), g);

        // now we're going to actually check if our weights are converging on
        // where they're needed to predict
        for (auto i = 0; i < x_train.size(); i++)
        {
            neuron_y = forward_pass(x_train[i], neuron_y, neuron_weights);
            std::cout << "x1 = " << x_train[i][1];
            std::cout << ", x2 = " << x_train[i][2];
            std::cout << ", y = " << neuron_y[2] << std::endl;

            // if our y train is opposite our prediction, we need to run the training again.
            if (((y_train[i] < 0.5) && (neuron_y[2] >= 0.5)) || ((y_train[i] >= 0.5) && (neuron_y[2] < 0.5)))
            {
                all_correct = false;
            }
        }
    }

    // plt.show()
}
