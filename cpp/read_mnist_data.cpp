// import idx2numpy
// #define idx2_Implementation
// #include "idx2.hpp"
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// Where dez images
// idx2::params TRAIN_IMAGE_P;
// TRAIN_IMAGE_P.InputFile = "data/mnist/train-images.idx3-ubyte";
// TRAIN_IMAGE_P.InDir = "data/mnist/";

// idx2::idx2_file Idx2;
// idx2_CleanUp(Dealloc(&Idx2)); // clean up Idx2 automatically in case of error
// idx2::grid OutGrid = idx2::GetOutputGrid(Idx2, TRAIN_IMAGE_P);

// idx2::buffer OutBuf;               // buffer to store the output
// idx2_CleanUp(DeallocBuf(&OutBuf)); // deallocate OutBuf automatically in case of error
// idx2::AllocBuf(&OutBuf, idx2::Prod<idx2::i64>(idx2::Dims(OutGrid)) * idx2::SizeOf(Idx2.DType));
// idx2::Decode(&Idx2, TRAIN_IMAGE_P, &OutBuf);

// std::string TRAIN_IMAGE_FILENAME = "data/mnist/train-images.idx3-ubyte";
// std::string TRAIN_LABEL_FILENAME = "data/mnist/train-labels.idx1-ubyte";
// std::string TEST_IMAGE_FILENAME = "data/mnist/t10k-images.idx3-ubyte";
// std::string TEST_LABEL_FILENAME = "data/mnist/t10k-labels.idx1-ubyte";

// train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
// train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
// test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
// test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

// print('Dimensions of train images ', train_images.shape)
// print('Dimensions of train labels ', train_labels.shape)
// print('Dimensions of test images ', test_images.shape)
// print('Dimensions of test labels ', test_labels.shape)

// # Show one training example
// print('label for first training example ', train_labels[0])
// print('---beginning of pattern for first training example---')
// for line in train_images[0]:
//     for num in line:
//         if num > 0:
//             print('*', end = ' ')
//         else:
//             print(' ', end = ' ')
//     print('')
// print('---end of pattern for first training exaple---')

// None of this is me.  It's from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class MnistDataloader
{
private:
    // we can do better
    std::string IMAGE_FILENAME = "../data/mnist/train-images.idx3-ubyte";
    std::string LABEL_FILENAME = "../data/mnist/train-labels.idx1-ubyte";

public:
    void read_images_labels()
    {
        read_images_labels("", "");
    }

    void read_images_labels(std::string input_image_filename, std::string input_label_filename)
    {

        if (!input_image_filename.empty())
        {
            this->IMAGE_FILENAME = input_image_filename;
        }

        if (!input_label_filename.empty())
        {
            this->LABEL_FILENAME = input_label_filename;
        }

        // read from the labels file
        std::ifstream labels_file{LABEL_FILENAME, labels_file.binary | labels_file.in};
        if (!labels_file.is_open())
        {
            std::cout << "Failed to open " << LABEL_FILENAME << std::endl;
            return;
        }

        // big enough for our magic number, 8 bytes.
        int64_t magic_number;
        // reinterpret_cast is compile time, and can convert between pointer types.
        labels_file.read(reinterpret_cast<char *>(&magic_number), sizeof magic_number);

        // include gcc endian conversion with builtin_bswap32, since this is big endian
        // we should just do this from the get go, once
        // std::cout << "First 8 bytes of file " << __builtin_bswap32(magic_number) << std::endl;
        if (__builtin_bswap32(magic_number) != 2049)
        {
            std::cout << "Magic number mismatch. Expected 2049, but got " << magic_number << std::endl;
            return;
        }

        // to hold our labels, which are stored in 2 bytes
        std::vector<uint16_t> labels = {};
        uint16_t label;

        while (labels_file.get(reinterpret_cast<char *>(&label), sizeof label))
        {
            labels.push_back(label);
        }

        labels_file.close();

        // read from the images file
        std::ifstream images_file{IMAGE_FILENAME, images_file.binary | images_file.in};
        if (!images_file.is_open())
        {
            std::cout << "Failed to open " << IMAGE_FILENAME << std::endl;
            return;
        }

        // our magic number, 4 bytes.
        int32_t images_magic_number;

        // reinterpret_cast is compile time, and can convert between pointer types.
        images_file.read(reinterpret_cast<char *>(&images_magic_number), sizeof images_magic_number);

        // check the first two bytes of the magic number, should always be 00
        if ((images_magic_number & 0x0000ffff) != 0)
        {
            std::cout << "Incorrect bytes in magic number for images file. Should be 0, but is " << (images_magic_number & 0xffff) << std::endl;
        }

        // include gcc endian conversion with builtin_bswap32, since this is big endian
        // std::cout << "First " << sizeof(images_magic_number) << " bytes of file " << __builtin_bswap32(images_magic_number) << std::endl;
        if (__builtin_bswap32(images_magic_number) != 2051)
        {
            std::cout << "Magic number mismatch. Expected 2051, but got " << __builtin_bswap32(images_magic_number) << std::endl;
            return;
        }

        // third byte is the data type. We really need to get this...
        std::cout << "Datatype " << (__builtin_bswap32(images_magic_number) & 0x0000ff00) << std::endl;

        // fourth byte is the number of dimensions
        // we can make the rest of below more generic by using this to get the sizes of each dimension
        std::cout << "Number of dimensions " << (__builtin_bswap32(images_magic_number) & 0x000000ff) << std::endl;

        // now let's get the row/col data
        uint32_t images_rows;
        images_file.read(reinterpret_cast<char *>(&images_rows), sizeof images_rows);
        std::cout << "Number of rows " << __builtin_bswap32(images_rows) << std::endl;

        uint32_t images_cols;
        images_file.read(reinterpret_cast<char *>(&images_cols), sizeof images_cols);
        std::cout << "Number of columns " << __builtin_bswap32(images_cols) << std::endl;

        uint32_t images_depth;
        images_file.read(reinterpret_cast<char *>(&images_depth), sizeof images_depth);
        std::cout << "Depth of data " << __builtin_bswap32(images_depth) << std::endl;

        auto images_full_data_size = images_rows * images_cols * images_depth;

        // to hold our images, which are stored in 2 bytes
        // std::vector<uint16_t> images = {};
        // uint16_t image;

        // while (images_file.get(reinterpret_cast<char *>(&image), sizeof image))
        // {
        //     labels.push_back(label);
        // }

        //             image_data = array("B", file.read())
        //         images = []
        //         for i in range(size):
        //             images.append([0] * rows * cols)
        //         for i in range(size):
        //             img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        //             img = img.reshape(28, 28)
        //             images[i][:] = img

        labels_file.close();

        //         return images, labels
    }
};

//     def read_images_labels(self, images_filepath, labels_filepath):
//         labels = []
//         with open(labels_filepath, 'rb') as file:
//             magic, size = struct.unpack(">II", file.read(8))
//             if magic != 2049:
//                 raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
//             labels = array("B", file.read())

//         with open(images_filepath, 'rb') as file:
//             magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
//             if magic != 2051:
//                 raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
//             image_data = array("B", file.read())
//         images = []
//         for i in range(size):
//             images.append([0] * rows * cols)
//         for i in range(size):
//             img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
//             img = img.reshape(28, 28)
//             images[i][:] = img

//         return images, labels

//     def load_data(self):
//         x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
//         x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
//         return (x_train, y_train),(x_test, y_test)

int main()
{
    MnistDataloader mnistDL;
    mnistDL.read_images_labels();
    std::cout << "blah"
              << "blah" << std::endl;

    return 1;
}
