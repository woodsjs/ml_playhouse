import idx2numpy

# Where dez images
TRAIN_IMAGE_FILENAME = 'data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = 'data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = 'data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = 'data/mnist/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

print('Dimensions of train images ', train_images.shape)
print('Dimensions of train labels ', train_labels.shape)
print('Dimensions of test images ', test_images.shape)
print('Dimensions of test labels ', test_labels.shape)

# Show one training example
print('label for first training example ', train_labels[0])
print('---beginning of pattern for first training example---')
for line in train_images[0]:
    for num in line:
        if num > 0:
            print('*', end = ' ')
        else:
            print(' ', end = ' ')
    print('')
print('---end of pattern for first training exaple---')