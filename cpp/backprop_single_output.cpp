// Once again, we're writing cpp based on python.
// What you looking at?

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

// # Randomly initialize neuron weights from -1.0 to 1.0
// # This also leaves the bias weight (at pos 0), as a zero
// # We want to randomly intitialize to break up the symmetry. If the
// # weights are all the same number, they would all get the same adjustment during backprop.
// def init_neuron_weights(input_count):
std::vector<double> init_neuron_weights(int input_count) {
    std::vector<double> weights(0.0, input_count+1);

    for (auto i = 1; i <= input_count+1; i++){
        weights[i] = 0; // needs to be some rando thing
        //         weights[i] = np.random.uniform(-1.0, 1.0)
    }

    return weights;
}

void show_learning(std::vector<double> weights){
    for(auto [w, i] = std::tuple{weights.begin(), 0}; w != weights.end(); w++, i++) {
        std::cout << "neuron " << i << " w0 = " << w[0] << " w1= " << w[1] << " w[2] = " << w[2] << std::endl;
    }
}

// # At the moment this is a two layer, three neuron network
// # Layer one, the hidden layer, is two neurons.  Layer two is a single neuron output layer
// def forward_pass(x):
//     global neuron_y

//     # first layer result is the tanh of our weights and input values
//     neuron_y[0] = np.tanh(np.dot(neuron_weights[0], x))
//     neuron_y[1] = np.tanh(np.dot(neuron_weights[1], x))

//     # this is really just to make it clear what our second neuron input looks like
//     neuron2_inputs = np.array([1.0, neuron_y[0], neuron_y[1]])

//     # This doesn't feel good, but OK let's just brute our second layer at the same time
//     # optimization would be to specify the number of layers and neurons per layer, like a good 
//     # lib does for us, but the goal here IS to see it clearly.  Sometimes that means coding
//     # it long form, then refactor
//     z2 = np.dot(neuron_weights[2], neuron2_inputs)

//     neuron_y[2] = 1.0/(1.0+np.exp(-z2))

// # now let's adjust our weights based on the error
// def backward_pass(y_truth):
//     global neuron_error

//     # your loss function derivative, from the output layer
//     error_prime = -(y_truth - neuron_y[2])
//     # and the logistic derivative. Remember, the derivatives build on each other through the layers
//     derivative = neuron_y[2] * (1.0 - neuron_y[2])
//     neuron_error[2] = error_prime * derivative

//     # tanh derivative, for the hidden layer
//     derivative = 1.0 - neuron_y[0]**2
//     neuron_error[0] = neuron_weights[2][1] * neuron_error[2] * derivative

//     derivative = 1.0 - neuron_y[1]**2
//     neuron_error[1] = neuron_weights[2][2] * neuron_error[2] * derivative

// def adjust_weights(x):
//     global neuron_weights

//     neuron_weights[0] -= (x * LEARNING_RATE * neuron_error[0])
//     neuron_weights[1] -= (x * LEARNING_RATE * neuron_error[1])

//     n2_inputs = np.array([1.0, neuron_y[0], neuron_y[1]])
//     neuron_weights[2] -= (n2_inputs * LEARNING_RATE * neuron_error[2])

int main()
{
    // # define our hyperparams
    // std::random_device rd;
    // std::mt19937 mt(rd());
    // std::uniform_distribution<int> dist(0, 6);
    // then we use dist(mt) to get our number

    // random.seed(7)
    float LEARNING_RATE = 0.1;
    std::vector<int> index_list = {0, 1, 2, 3}; // will be used to randomize order

    // training examples.Ultimately, like the hyperparams, these should be passed in.
    // here we include in code as we write it
    // First element in the vector v must be 1, our Bias value
    // Len of w, x is n+1 for n inputs
    

    std::vector<std::vector<double>> x_train = {{1.0, -1.0, -1.0},
                                                {1.0, -1.0, 1.0},
                                                {1.0, 1.0, -1.0},
                                                {1.0, 1.0, 1.0}};
    std::vector<double> y_train = {0.0, 1.0, 1.0, 0.0}; // ground truth

    // define perceptron weights. THese are arbitrarily chosen.
    std::vector<double> weights = {0.2, -0.6, 0.25};


    std::vector<std::vector<double>> neuron_weights = {init_neuron_weights(2), init_neuron_weights(2), init_neuron_weights(2)};
    std::vector<double> neuron_y(0.0,3);
    std::vector<double> neuron_error(0.0,3);
    // print initial weights
    show_learning(weights);

    // This is the meat and potatoes Perceptron training loop
    bool all_correct = false;
        
    while (!all_correct)
    {
        all_correct = true;

        for (auto i : index_list)
        {
//     # take the randomized list vals in
//     for i in index_list:
//         forward_pass(x_train[i])
//         backward_pass(y_train[i])
//         adjust_weights(x_train[i])

//         show_learning()
        // TODO: get that rand working
        // random.shuffle(index_list)

        // take the randomized list vals in
        }

        for ( auto i=0; i < x_train.size(); i++)
        {
        //     for i in range(len(x_train)):
//         forward_pass(x_train[i])
//         print('x1 = ', '%4.1f' % x_train[i][1],
//               ', x2 = ', '%4.1f' % x_train[i][2],
//               ', y = ',  '%4.1f' % neuron_y[2])
        
//         if(((y_train[i] < 0.5) and (neuron_y[2] >= 0.5))
//            or ((y_train[i] >= 0.5) and (neuron_y[2] < 0.5))):
//             all_correct = False
        }
    }

    // plt.show()
}
