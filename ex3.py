import numpy as np
import matplotlib.pyplot as plt

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
        print(f"n={result['n']}, m={result['m']}")
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


