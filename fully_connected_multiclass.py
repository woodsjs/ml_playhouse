import idx2numpy

TRAIN_IMAGE_FILENAME = './data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = './data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = './data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = './data/mnist/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

print(f'dimensions of train_images: {train_images.shape}')
print(f'dimensions of train_labels: {train_labels.shape}')
print(f'dimensions of test_images: {test_images.shape}')
print(f'dimensions of test_labels: {test_labels.shape}')