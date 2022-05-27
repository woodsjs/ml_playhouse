# simple perceptron
# Element 1 in X must be 1
# w and x must be len n+1 for n inputs
import random

def show_learning(w):
    print(f'w0  {w[0]:.2f}, w1 = {w[1]:.2f}, w2 = {w[2]:.2f}')

# Define basic vars
random.seed(7)
LEARNING_RATE = 0.1
index_list = [0,1,2,3]

# x and y train
X_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]

# weights
w = [0.2, -0.6, 0.25]

print(f'Initial weights: {show_learning(w)}')

def compute_output(w, x):
    z = 0.0

    for i in range(len(w)):
        z += x[i] * w[i]

    if z < 0:
        return -1
    else:
        return 1

all_correct = False

while not all_correct:
    all_correct = True
    random.shuffle(index_list)

    for i in index_list:
        x = X_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)

        if y != p_out:
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False

            show_learning(w)


#w = [.02, .1, .232]
#x = [1, 2, 3]

#print(compute_output(w, x))

