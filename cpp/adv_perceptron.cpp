#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

// hey, don't judge!
//  1. We'd probably use more goodies from std:: here but we're not (yet?)
//  2. We're ultimately taking our python code and brute force swapping it to c++

// we're not plotting here yet.
// Yeah, the commented stuff is Python. You gotta problem with that?
// # plot vars
// color_list = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
// color_index = 0

// TODO: pass by ref
void show_learning(std::vector<double> w)
{
    //     global color_index
    //     print('w0 = ', '%5.2f' % w[0], ', w1 = ' , '%5.2f' %w[1], ', w2 = ', '%5.2f' %w[2])
    std::cout << "w0 = " << w[0] << " w1 = " << w[1] << " w2 = " << w[2] << std::endl;
    //     if color_index == 0:
    //         plt.plot([1.0], [1.0], 'b_', markersize=12)
    //         plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
    //         plt.axis([-2, 2, -2, 2])
    //         plt.xlabel('x1')
    //         plt.ylabel('x2')

    //     x = [-2.0, 2.0]

    //     if abs(w[2]) < 1e-5:
    //         y = [-w[1]/(1e-5)*(-2.0)+(-w[0]/(1e-5)),
    //              -w[1]/(1e-5)*((2.0)+(-w[0]/(1e-5)))]
    //     else:
    //         y = [-w[1]/w[2]*(x[0])+(-w[0]/w[2]),
    //              -w[1]/w[2]*(x[1])+(-w[0]/w[2])]

    //     plt.plot(x, y, color_list[color_index])

    //     if color_index < (len(color_list) - 1):
    //         color_index += 1
}

int compute_output(std::vector<double> w, std::vector<double> v)
{
    std::vector<double> mult_intermediate(w.size(), 0);

    // fun with numbers. We'd use boost here for some goodness?
    std::transform(v.begin(), v.end(), w.begin(), mult_intermediate.begin(), [](auto x, auto y){ return x * y; });
    auto z = std::reduce(mult_intermediate.begin(), mult_intermediate.end(), 0.0, [](auto x, auto y){return(x+y);});

    // signum function
    if (z < 0)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

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
    std::vector<double> y_train = {1.0, 1.0, 1.0, -1.0}; // ground truth

    // define perceptron weights. THese are arbitrarily chosen.
    std::vector<double> w = {0.2, -0.6, 0.25};

    // print initial weights
    show_learning(w);

    // This is the meat and potatoes Perceptron training loop
    bool all_correct = false;

    while (!all_correct)
    {
        all_correct = true;

        // TODO: get that rand working
        // random.shuffle(index_list)

        // take the randomized list vals in
        for (auto i : index_list)
        {
            auto x = x_train[i];
            auto y = y_train[i];

            // run the perceptron function on each list item
            auto p_out = compute_output(w, x);

            // our output doesn't match our ground truth
            if (y != p_out) // we're stopping at five for testing
            {
                // we're going to adjust the weights
                for (auto j = 0; j < w.size(); j++)
                {
                    // take a weight from the list and add it to
                    // our ground truth, multiplied by the learning rate, multiplied by
                    // our matching input value
                    // This moves our weight up or down, based on the ground truth's positive
                    // or negative value
                    w[j] += (y * LEARNING_RATE * x[j]);
                }
                all_correct = false;
                show_learning(w);
            }
        }
    }

    // plt.show()
}
