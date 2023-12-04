#include <iostream>
#include <vector>

// first element in v vector must be 1
// length of w and v must be n+1 for n len neuron

int compute_output(std::vector<double> w, std::vector<double> v)
{
    double z = 0.0;

    for (size_t i = 0; i < w.size(); i++)
    {
        z += v[i] * w[i];
    }

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
    std::vector<double> w{0.9, -0.6, -0.5};
    std::vector<double> v{1.0, -1.0, -1.0};
    std::vector<double> v2{1.0, -1.0, 1.0};
    std::vector<double> v3{1.0, 1.0, -1.0};
    std::vector<double> v4{1.0, 1.0, 1.0};

    std::cout << compute_output(w, v) << std::endl;
    std::cout << compute_output(w, v2) << std::endl;
    std::cout << compute_output(w, v3) << std::endl;
    std::cout << compute_output(w, v4) << std::endl;
}
