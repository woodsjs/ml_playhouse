# started as perceptron.py, added niceness
# part of this is the use of numpy

# Element 1 in X must be 1
# w and x must be len n+1 for n inputs
import matplotlib.pyplot as plt
# import matplotlib.random as random
import random
import numpy as np1

# Define basic vars
random.seed(7)
LEARNING_RATE = 0.1
index_list = [0,1,2,3]

color_list = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
color_index = 0

# x and y train
X_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]

# weights
w = [0.2, -0.6, 0.25]

# weights as given for a multi perceptron. In this case, with two neurons
wm = [[0.2, -0.6, 0.25], [0.23, -0.62, 0.27]]

#somewhere to store out plot
final_plot = None

def show_learning(w):
    """
        if we're command line only, print some numbers and be gone.
        
        Otherwise show some pretty pictures, plot the regression lines
    """

    global color_index

    print(f'w0  {w[0]:.2f}, w1 = {w[1]:.2f}, w2 = {w[2]:.2f}')

    # First run through, so set up the plot. This includes adding '-' and '+' at our data points
    # the datapoints come from X_train, here they are hard coded
    if color_index == 0:
        plt.plot([1.0], [1.0], 'b_', markersize=12)
        plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
        plt.axis([-2,2,-2,2])
        plt.xlabel('x1')
        plt.ylabel('x2')
    
    x = [-2.0, 2.0]

    # Why weight two? Who frikken knows
    # We're just looking if it's super close to zero (lt 0.00001) if it is, we use the floor of 1e-5
    # for the denominator
    #
    # we're looking at the reality that if our weighted sum of inputs is lt or gt 0
    # so let's avoid divide by 0 errors, k?
    if abs(w[2]) < 1e-5:
        y = [-w[1]/(1e-5)*(-2.0)+(w[0]/(1e-5)),
            -w[1]/(1e-5)*(2.0)+(-w[0]/(1e-5))]
    else:
        y = [-w[1]/w[2]*(-2.0)+(w[0]/(w[2])),
             -w[1]/(w[2])*(2.0)+(-w[0]/(w[2]))]

    plt.plot(x, y, color_list[color_index])

    if color_index < (len(color_list) -1):
        color_index += 1

    if color_index == len(X_train):
        plt.show()

# print(f'Initial weights: {show_learning(w)}')

def compute_output(w, x):
    z = 0.0

    for i in range(len(w)):
        z += x[i] * w[i]

    if z < 0:
        return -1
    else:
        return 1

# matrix version of compute_output
def compute_output_vector(w, x):
    z = np.dot(w, x)
    return np.sign(z)

all_correct = False

while not all_correct:
    all_correct = True
    random.shuffle(index_list)

    for i in index_list:
        x = X_train[i]
        y = y_train[i]

        # p_out = compute_output(w, x)
        p_out = compute_output_vector(w, x)

        # checking our output vs y_train
        if y != p_out:
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False

            show_learning(w)

#w = [.02, .1, .232]
#x = [1, 2, 3]

#print(compute_output(w, x))

