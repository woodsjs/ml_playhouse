#ifndef MNISTDATALOADER_H // include guard
#define MNISTDATALOADER_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

class MnistDataloader
{
private:
    // we can do better
    std::string IMAGE_FILENAME = "../data/mnist/train-images.idx3-ubyte";
    std::string LABEL_FILENAME = "../data/mnist/train-labels.idx1-ubyte";

public:
    MnistDataloader() = default;
    ~MnistDataloader(){};

    void read_images_labels(std::string input_image_filename, std::string input_label_filename, std::vector<uint16_t> &labels, std::vector<std::vector<std::vector<uint8_t>>> &images);
    void read_images_labels(std::vector<uint16_t> &labels, std::vector<std::vector<std::vector<uint8_t>>> &images);
};

#endif