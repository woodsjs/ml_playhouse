#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

// None of this is me.  It's from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class MnistDataloader
{
private:
    // we can do better
    std::string IMAGE_FILENAME = "../data/mnist/train-images.idx3-ubyte";
    std::string LABEL_FILENAME = "../data/mnist/train-labels.idx1-ubyte";

public:
    void read_images_labels(std::vector<uint16_t> &labels, std::vector<std::vector<std::vector<uint8_t>>> images)
    {
        read_images_labels("", "", labels, images);
    }

    void read_images_labels(std::string input_image_filename, std::string input_label_filename, std::vector<uint16_t> &labels, std::vector<std::vector<std::vector<uint8_t>>> &images)
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
        // std::vector<uint16_t> labels = {};
        uint16_t label;

        while (labels_file.get(reinterpret_cast<char *>(&label), sizeof label))
        {
            labels.push_back(label);
        }

        labels_file.close();

        // read from the images file
        // https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
        std::ifstream images_file{IMAGE_FILENAME, std::ios::binary | images_file.in};
        if (!images_file.is_open())
        {
            std::cout << "Failed to open " << IMAGE_FILENAME << std::endl;
            return;
        }

        // our magic number, 4 bytes.
        int32_t images_magic_number;

        // reinterpret_cast is compile time, and can convert between pointer types.
        // The sizes in each dimension are 4-byte integers (big endian, like in most non-Intel processors).
        // so later we use the builtin_bswap32 to flipitty do
        images_file.read(reinterpret_cast<char *>(&images_magic_number), sizeof images_magic_number);

        int32_t bswapped_images_magic_number = __builtin_bswap32(images_magic_number);

        // check the first two bytes of the magic number, should always be 00
        if ((bswapped_images_magic_number & 0xffff0000) != 0)
        {
            std::cout << "Incorrect bytes in magic number for images file. Should be 0, but is " << (images_magic_number & 0xffff) << std::endl;
        }

        // make sure the magic number is good
        if (bswapped_images_magic_number != 2051)
        {
            std::cout << "Magic number mismatch. Expected 2051, but got " << bswapped_images_magic_number << std::endl;
            return;
        }

        // third byte is the data type. We really need to get this...
        // let's just check that it's 0800 hex, which is a byte
        // then why not NOT do anything with it..lul
        std::stringstream stream;
        stream << std::setfill('0') << std::setw(sizeof(uint16_t) * 2)
               << std::hex << (bswapped_images_magic_number & 0x0000ff00);
        std::string datatype(stream.str());

        auto datasize = 1;
        if (datatype != "0800")
        {
            std::cout << "Datatype is not a byte sized one." << std::endl;
        }

        // fourth byte is the number of dimensions
        // we can make the rest of below more generic by using this to get the sizes of each dimension
        // now let's get the overall size
        uint32_t num_images;
        // images_file.read(reinterpret_cast<char *>(&num_images), sizeof num_images);
        images_file.read(reinterpret_cast<char *>(&num_images), sizeof num_images);
        uint32_t bswapped_num_images = __builtin_bswap32(num_images);

        // col size
        uint32_t num_image_cols;
        images_file.read(reinterpret_cast<char *>(&num_image_cols), sizeof num_image_cols);
        uint32_t bswapped_num_image_cols = __builtin_bswap32(num_image_cols);

        // rows in each image
        uint32_t num_image_rows;
        images_file.read(reinterpret_cast<char *>(&num_image_rows), sizeof num_image_rows);
        uint32_t bswapped_num_image_rows = __builtin_bswap32(num_image_rows);

        auto images_full_data_size = bswapped_num_image_cols * bswapped_num_image_rows * bswapped_num_images;

        // so our images are stored in 8 byte chunks, 28 rows x 28 cols for each image
        uint8_t value;
        std::vector<uint8_t> row;
        std::vector<std::vector<uint8_t>> image;
        // std::vector<std::vector<std::vector<uint8_t>>> images;

        for (auto h = 0; h < bswapped_num_images; h++)
        {
            image.clear();
            for (auto i = 0; i < bswapped_num_image_rows; i++)
            {
                row.clear();
                for (auto j = 0; j < bswapped_num_image_cols; j++)
                {
                    images_file.read(reinterpret_cast<char *>(&value), sizeof value);

                    row.push_back(value);
                }
                image.push_back(row);
            }
            images.push_back(image);
        }

        std::cout << "Label for training example " << labels[0] << std::endl;
        std::cout << "---beginning of pattern for training example---" << std::endl;
        for (auto row : images[0])
        {
            for (auto num : row)
            {
                std::cout << (num > 0 ? "*" : " ") << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "---end of pattern for training example---" << std::endl;

        images_file.close();

        //         return images, labels
    }
};

//     def load_data(self):
//         x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
//         x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
//         return (x_train, y_train),(x_test, y_test)

int main()
{
    MnistDataloader mnistDL;
    std::vector<uint16_t> labels;
    std::vector<std::vector<std::vector<uint8_t>>> images;

    mnistDL.read_images_labels(labels, images);

    return 1;
}
