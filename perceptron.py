# simple perceptron
# Element 1 in X must be 1
# w and x must be len n+1 for n inputs

def compute_output(w, x):
    z = 0.0

    for i in range(len(w)):
        z += x[i] * w[i]
    if z < 0:
        return -1
    else:
        print(f'Z was {z}')
        return 1

w = [.02, .1, .232]
x = [1, 2, 3]

print(compute_output(w, x))

