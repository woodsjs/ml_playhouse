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
#include <numeric>

// np.random.seed(7)
const float LEARNING_RATE = 0.01;
const int EPOCHS = 5;

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
    std::vector<std::vector<double>> weights(neuron_count, std::vector<double>(input_count + 1, 0.0));
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> distDoub(-1.0, 1.0); // distribution in range [-1.0, 1.0];

    for (auto i = 0; i < neuron_count; i++)
    {
        for (auto j = 1; j < input_count; j++)
            weights[i][j] = distDoub(rng); // needs to be some rando thing
    }

    return weights;
}

// text representation of learning during one epoch
// we store the data in global vars to use later
void show_learning(int epoch_number, double train_acc, double test_acc)
{
    // global chart_x
    // global chart_y_train
    // global chart_y_test

    std::cout << "Epoch no: " << epoch_number << " ";
    std::cout << ", train_acc: " << train_acc << " ";
    std::cout << ", test_acc " << test_acc << " ";

    // chart_x.append(epoch_no + 1)
    // chart_y_train.append(1.0 - train_acc)
    // chart_y_test.append(1.0 - test_acc)}
}

// def plot_learning():
//     plt.plot(chart_x, chart_y_train, "r-", label="training error")
//     plt.plot(chart_x, chart_y_test, "b-", label="test error")

//     plt.axis([0, len(chart_x), 0, 1.0])

//     plt.xlabel("training epochs")
//     plt.ylabel("error")

//     plt.legend()

//     plt.show()

// we pass in a single image and run it through all of our neurons
void forward_pass(std::vector<double> &x, std::vector<double> &hidden_layer_y, std::vector<double> &output_layer_y, std::vector<std::vector<double>> &hidden_layer_weights, std::vector<std::vector<double>> &output_layer_weights)
{

 
    for (auto i = 0; i < hidden_layer_weights.size(); i++)
    {
		double z;

		for ( auto j = 0; j < hidden_layer_weights[i].size(); j++) {
			z += x[i]*hidden_layer_weights[i][j];
		}
      	
		hidden_layer_y[i] = std::tanh(z);
    }


    // add in our bias term!
    // show the output of each hidden layer, to each input of the output layer
    std::vector<double> hidden_output_array = {1.0};
    hidden_output_array.insert(hidden_output_array.end(), hidden_layer_y.begin(), hidden_layer_y.end());


    for (auto i = 0; i < output_layer_weights.size(); i++)
    {
	
		double z;
		for ( auto j = 0; j < output_layer_weights[i].size(); j++) {
		  z += hidden_output_array[i]*output_layer_weights[i][j];
		}
 
	output_layer_y[i] = 1.0 / (1.0 + std::exp(-z));
    }
}

void backward_pass(std::vector<uint16_t> &y_truth, std::vector<double> &hidden_layer_error, std::vector<double> &output_layer_error, std::vector<double> &output_layer_y, std::vector<double> &hidden_layer_y, std::vector<std::vector<double>> &output_layer_weights)
{

    //     # backproping error
    //     # starting backwards, compute derivative of loss function for each output neuron
    for (auto i = 0; i < output_layer_y.size(); i++)
    {
        double error_prime = -(y_truth[i] - output_layer_y[i]);            // loss derivative
        double derivative = output_layer_y[i] * (1.0 - output_layer_y[i]); // logistic derivative

        output_layer_error[i] = error_prime * derivative;
    }

    for (auto i = 0; i < hidden_layer_y.size(); i++)
    {
        double error_weights;

        double derivative = 1.0 - pow(hidden_layer_y[i], 2); 
							    
		for ( auto j = 0; j < output_layer_weights.size(); j++ ) {
			//	take each hidden neurons error, total of 25 
			//	that equals the amtching output layer weights, total of 10 of 25 
			error_weights += output_layer_weights[j][i+1] * output_layer_error[j];
		}

        hidden_layer_error[i] = error_weights * derivative;
			
    }
}

void adjust_weights(std::vector<double> &x, std::vector<double> &hidden_layer_error, std::vector<double> &output_layer_error, std::vector<std::vector<double>> &hidden_layer_weights, std::vector<double> &hidden_layer_y, std::vector<std::vector<double>> &output_layer_weights)
{
    double local_learning_rate = LEARNING_RATE;

    for ( auto i = 0; i < hidden_layer_error.size(); i++) {

	    for ( auto j = 0; j < hidden_layer_weights[i].size(); j++) {
		    hidden_layer_weights[i][j] -= ( x[j] * LEARNING_RATE * hidden_layer_error[i]);
	    }
    }
    // add intermediate results destroy

    // add in our bias term!
    //std::vector<std::vector<double>> hidden_output_array(hidden_layer_y.size(), std::vector<double>(2,0.0));
    //std::transform(hidden_layer_y.begin(), hidden_layer_y.end(), hidden_output_array.begin(),[](auto x) -> std::vector<double> {return {1.0, x};});

   for ( auto i = 0; i < output_layer_error.size(); i++ ) { // 10 rows
	   for ( auto j = 0; j < output_layer_weights[i].size(); j++) { // 10 rows of 25 cols
			// hidden output array is 25 rows, 2 cols
			// output_layer_error is 10 rows
		
		   output_layer_weights[i][j] -= ( 1.0 * LEARNING_RATE * output_layer_error[i]);
		   output_layer_weights[i][j] -= ( hidden_layer_y[j] * LEARNING_RATE * output_layer_error[i]);
	   }
   }

}

int main(void)
{
	auto hidden_neuron_count = 25;
	auto hidden_layer_inputs = 784;

    auto output_neuron_count = 10;

    // our mnist reader will push out our labels (y_train, y_test), as well as our images (x_train, x_test)
    std::vector<uint16_t> y_train;
    std::vector<std::vector<std::vector<uint8_t>>> x_train;
    
    std::vector<uint16_t> y_test;
    std::vector<std::vector<std::vector<uint8_t>>> x_test;

    read_mnist(TRAIN_IMAGE_FILENAME, TRAIN_LABEL_FILENAME, y_train, x_train);
    read_mnist(TEST_IMAGE_FILENAME, TEST_LABEL_FILENAME, y_test, x_test);

    // flatten the x_train, x_tests
    std::vector<std::vector<uint8_t>> flat_x_train;
   
    for (auto i = 0; i < x_train.size(); i++)
    {
	std::vector<uint8_t> temp_image;

	for (auto j = 0; j < x_train[i].size(); j++){	
            temp_image.insert(temp_image.end(), x_train[i][j].begin(), x_train[i][j].end());
	}
	flat_x_train.push_back(temp_image);
    }

    std::vector<std::vector<uint8_t>> flat_x_test;
   
    for (auto i = 0; i < x_test.size(); i++)
    {
	std::vector<uint8_t> temp_image;

	for (auto j = 0; j < x_test[i].size(); j++){	
            temp_image.insert(temp_image.end(), x_test[i][j].begin(), x_test[i][j].end());
	}
	flat_x_test.push_back(temp_image);
    }

    // Set up the ability to normalize the data
    std::vector<uint8_t> super_flat;
    for (auto i = 0; i < flat_x_train.size(); i++) {
	   super_flat.insert(super_flat.end(), flat_x_train[i].begin(), flat_x_train[i].end());
    }

    double sum = std::accumulate(super_flat.begin(), super_flat.end(), 0.0);
    double mean =  sum / super_flat.size();

    double accum = 0.0;
    std::for_each (super_flat.begin(), super_flat.end(), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });

    double stdev = sqrt(accum / (super_flat.size()-1));


    std::vector<std::vector<double>> x_train_normalized(flat_x_train.size(),std::vector<double>(flat_x_train[0].size())); //don't do this
    for (auto i = 0; i < flat_x_train.size(); i++) {
	    std::transform(flat_x_train[i].begin(), flat_x_train[i].end(), x_train_normalized[i].begin(), [&](auto x){return (x - mean)/stdev;});
    }

    std::vector<std::vector<double>> x_test_normalized(flat_x_test.size(),std::vector<double>(flat_x_test[0].size())); //don't do this
    for (auto i = 0; i < flat_x_test.size(); i++) {
	    std::transform(flat_x_test[i].begin(), flat_x_test[i].end(), x_test_normalized[i].begin(), [&](auto x){return (x - mean)/stdev;});
    }


    // index_list = list(range(len(x_train)))
    // std::vector<int> index_list; 

    // get our weights
    // std::vector<std::vector<double>> neuron_weights;
    // neuron_weights = init_neuron_weights(3, 24);

    // # 25 neurons, 784 inputs for our hidden layer
    std::vector<std::vector<double>> hidden_layer_weights = init_neuron_weights(hidden_neuron_count, hidden_layer_inputs);
    // 25neurons gives us 25 outputs, init to zero
    std::vector<double> hidden_layer_y(hidden_neuron_count, 0.0);
    // also, 25 neurons, 25 error values, init to zero
    std::vector<double> hidden_layer_error(hidden_neuron_count, 0.0);

    // the hidden layer has 25 outputs, so the output layer has 25 inputs
    // it has 10 outputs, since we're determining values 0-9
    std::vector<std::vector<double>> output_layer_weights = init_neuron_weights(output_neuron_count, hidden_neuron_count);
    std::vector<double> output_layer_y(output_neuron_count, 0.0);
    std::vector<double> output_layer_error(output_neuron_count, 0.0);

    // # here we're going to show our learning
    // chart_x = []
    // chart_y_train = []
    // chart_y_test = []

    // # training loop
	for (auto i = 0; i < EPOCHS; i++)
	{
        std::cout << "Epoch " << i << std::endl;

        // np.random.shuffle(index_list)
        auto correct_training_results = 0;

        // # don't really like that we're using global vars
        // # below doesn't give a good feel as to what's going on for a programmer
        std::cout << "Training set size " << x_train_normalized.size() << std::endl;
        for (auto j = 0; j < x_train_normalized.size(); j++)
        {
            if (j % 10000 == 0)
            {
                std::cout << " " << j << " ";
            }
            else if (j % 1000 == 0)
            {
                std::cout << "*";
            }
	    
            forward_pass(x_train_normalized[j], hidden_layer_y, output_layer_y, hidden_layer_weights, output_layer_weights);

            std::vector<double>::iterator result;
            result = std::max_element(output_layer_y.begin(), output_layer_y.end());
            int max_y = std::distance(output_layer_y.begin(), result);

            if (max_y == y_train[j])
            {
                correct_training_results += 1;
            }
            backward_pass(y_train, hidden_layer_error, output_layer_error, output_layer_y, hidden_layer_y, output_layer_weights);

            adjust_weights(x_train_normalized[j], hidden_layer_error, output_layer_error, hidden_layer_weights, hidden_layer_y, output_layer_weights);

        }
        std::cout << std::endl;
        auto correct_test_results = 0;
        for (auto k = 0; k < x_test_normalized.size(); k++)
        {

            forward_pass(x_test_normalized[k], hidden_layer_y, output_layer_y, hidden_layer_weights, output_layer_weights);

            std::vector<double>::iterator result;
            result = std::max_element(output_layer_y.begin(), output_layer_y.end());
            int max_y = std::distance(output_layer_y.begin(), result);

            if (max_y == y_test[k])
            {
                correct_test_results += 1;
            }
        }

        std::cout << "Xtrain size " << x_train_normalized.size() << std::endl;
        double training_accuracy = static_cast<double>(correct_training_results) / static_cast<double>(x_train_normalized.size());
        double testing_accuracy = static_cast<double>(correct_test_results) / static_cast<double>(x_test_normalized.size());

        show_learning(i, training_accuracy, testing_accuracy);
        std::cout << std::endl;

        // plot_learning()
	}

    return EXIT_SUCCESS;
}
