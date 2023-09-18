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

# First element in the vector v must be 1, our Bias value
# Len of w, x is n+1 for n inputs

def compute_output(w, v):
    z = 0.0
    
    for i in range( len(w) ):
        z += v[i] * w[i] # weighted sum of inputs
    
    # signum function
    if z < 0: 
        return -1
    else:
        return 1
    
# This is the meat and potatoes
# Perceptron training loop
all_correct = False

while not all_correct:
    all_correct = True
    random.shuffle(index_list)

    # take the randomized list vals in
    for i in index_list:
        x = x_train[i]
        y = y_train[i]

        # run the perceptron function on each list item
        p_out = compute_output(w, x)
        
        # our output doesn't match our ground truth
        if y != p_out:
            # we're going to adjust the weights
            for j in range(0, len(w)):

                # take a weight from the list and add it to
                # our ground truth, multiplied by the learning rate, multiplied by 
                # our matching input value
                # This moves our weight up or down, based on the ground truth's positive
                # or negative value
                w[j] += (y * LEARNING_RATE * x[j])

            all_correct = False

            show_learning(w)

