import numpy as np

def linear_regression_gradient_descent(X, Y, k, t, learning_rate):
    # Step 1: Initialize parameters (beta) randomly
    m = X.shape[1]  # Number of features (including intercept)
    beta = np.random.randn(m)  # Random initialization of beta
    
    # Step 2: Initialize variables for Gradient Descent
    n = X.shape[0]  # Number of samples
    prev_cost = float('inf')  # Set previous cost to infinity initially
    final_cost = None  # To store the final cost function value
    
    # Step 3: Run Gradient Descent
    for iteration in range(k):
        # Step 3a: Calculate the predictions
        predictions = X.dot(beta)
        
        # Step 3b: Calculate the residuals/errors
        errors = predictions - Y.reshape(-1)  # Reshape Y to ensure proper subtraction
        
        # Step 3c: Calculate the cost function (Mean Squared Error)
        cost = (1/(2*n)) * np.sum(errors ** 2)
        
        # Step 3d: Calculate the gradient
        gradient = (1/n) * X.T.dot(errors)
        
        # Step 3e: Update the parameters (beta)
        beta = beta - learning_rate * gradient
        
        # Step 3f: Check for convergence (change in cost function)
        if abs(prev_cost - cost) < t:
            print(f"Converged after {iteration + 1} iterations.")
            final_cost = cost
            break
        
        # Update the previous cost
        prev_cost = cost
    
    # If the loop finishes without convergence, set the final cost
    if final_cost is None:
        final_cost = cost
    
    return beta, final_cost

# Example usage:
n = 100
m = 5
X = np.random.randn(n, m + 1)  # Including intercept term
Y = np.random.randn(n)  # Random Y values

k = 1000         # Number of iterations
t = 1e-6         # Threshold for cost function change
learning_rate = 0.01  # Learning rate

beta, final_cost = linear_regression_gradient_descent(X, Y, k, t, learning_rate)
print("Learned coefficients (beta):", beta)
print("Final cost function value:", final_cost)

#---OUTPUT---------
#Converged after 798 iterations.
#Learned coefficients (beta): [ 0.02747227  0.0075521  -0.12572149 -0.12125614 -0.10061729  0.16750731]
#Final cost function value: 0.4315146109282243
