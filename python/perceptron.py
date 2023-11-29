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

print(compute_output( [ 0.9, -0.6, -0.5], [1.0, -1.0, -1.0] ))
print(compute_output( [ 0.9, -0.6, -0.5], [1.0, -1.0, 1.0] ))
print(compute_output( [ 0.9, -0.6, -0.5], [1.0, 1.0, -1.0] ))
print(compute_output( [ 0.9, -0.6, -0.5], [1.0, 1.0, 1.0] ))