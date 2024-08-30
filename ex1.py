# ----------------------Ques 1------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(sigma, n, m):
    # Step 1: Generate random independent variables X
    X = np.random.randn(n, m)
    
    # Step 2: Add the intercept term (xi0 = 1 for all i)
    X = np.hstack((np.ones((n, 1)), X))  # Add a column of ones for the intercept
    
    # Step 3: Generate random coefficients β (of dimensionality m + 1)
    beta = np.random.randn(m + 1)
    
    # Step 4: Generate Gaussian noise e with mean 0 and standard deviation sigma
    e = np.random.normal(0, sigma, n)
    
    # Step 5: Calculate the dependent variable Y
    Y = X.dot(beta) + e
    
    # Return X, Y, and β
    return X, Y, beta

# Example usage:
sigma = 2.0 # Standard deviation of Gaussian noise 
n = 50      # Number of samples
m = 3      # Number of independent variables

X, Y, beta = generate_dataset(sigma, n, m)
print("X:", X.shape)  # Output: X: (100, 6)
print("Y:", Y.shape)  # Output: Y: (100,)
print("beta:", beta.shape)  # Output:beta: (6,)

# ---------------------------------------------------------------------------


# -----------------------------Ques 2-----------------------------------------
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
            print(f"\nConverged after {iteration + 1} iterations.")
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

# ---------------------------------------------------------------------------

# ----------------------------Ques 3-------------------------------------------



def generate_data(n, m, sigma):
    beta_true = np.random.randn(m + 1)  # True coefficients
    X = np.random.randn(n, m + 1)
    X[:, 0] = 1  # Intercept term
    noise = np.random.normal(0, sigma, size=n)
    Y = X.dot(beta_true) + noise
    return X, Y, beta_true

def linear_regression_gradient_descent(X, Y, k, t, learning_rate):
    m = X.shape[1]
    beta = np.random.randn(m)
    n = X.shape[0]
    prev_cost = float('inf')
    
    for iteration in range(k):
        predictions = X.dot(beta)
        errors = predictions - Y
        cost = (1/(2*n)) * np.sum(errors ** 2)
        gradient = (1/n) * X.T.dot(errors)
        beta -= learning_rate * gradient
        
        if abs(prev_cost - cost) < t:
            return beta, cost, iteration + 1
        
        prev_cost = cost
    
    return beta, cost, k

def run_and_report(sigma, k, t, learning_rate):
    n_values = [50, 100]
    m_values = [2, 5]
    results = []

    for n in n_values:
        for m in m_values:
            X, Y, beta_true = generate_data(n, m, sigma)
            beta_learned, final_cost, iterations = linear_regression_gradient_descent(X, Y, k, t, learning_rate)
            
            results.append({
                "n": n,
                "m": m,
                "beta_true": beta_true,
                "beta_learned": beta_learned,
                "final_cost": final_cost,
                "iterations": iterations
            })
    
    for result in results:
        print(f" \n n={result['n']}, m={result['m']}")
        print(f"True Coefficients: {result['beta_true']}")
        print(f"Learned Coefficients: {result['beta_learned']}")
        print(f"Final Cost: {result['final_cost']}, Iterations: {result['iterations']}")
        print("=" * 50)
        
        plt.figure(figsize=(10, 5))
        plt.plot(result['beta_true'], 'o-', label='True Coefficients')
        plt.plot(result['beta_learned'], 'x--', label='Learned Coefficients')
        plt.title(f'n={result["n"]}, m={result["m"]}')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Coefficient Value')
        plt.legend()
        plt.show()

# Experiment parameters
sigma = 0.5
k = 1000
t = 1e-6
learning_rate = 0.01

# Run the experiment and report results
run_and_report(sigma, k, t, learning_rate)

