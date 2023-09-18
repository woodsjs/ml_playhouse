import random

def show_learning(w):
    print('w0 = ', '%5.2f' % w[0], ', w1 = ' , '%5.2f' %w[1], ', w2 = ', '%5.2f' %w[2])

# define our hyperparams
random.seed(7)
LEARNING_RATE = 0.1
index_list = [0,1,2,3] # will be used to randomize order

# training examples. Ultimately, like the hyperparams, these should be passed in.
# here we include in code as we write it
x_train = [(1.0, -1.0, -1.0),
           (1.0, -1.0, 1.0),
           (1.0, 1.0, -1.0),
           (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0] # ground truth

#define perceptron weights. THese are arbitrarily chosen.
w = [0.2, -0.6, 0.25]

#print initial weights
show_learning(w)

