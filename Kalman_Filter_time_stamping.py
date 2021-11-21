#%%
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.stats as stats
import time


#%%
# STATE EXTRAPOLATION EQUATION
def state_extrapolation(X_, F_):
    X_extr = F_.dot(X_)
    return X_extr

# COVARIANCE EXTRAPOLATION EQUATIONS
def cov_extrapolation(P_, Q_, F_):
    P_extr = F_.dot(P_).dot(F_.T) + Q_
    return P_extr

# KALMAN GAIN EQUATION
def calculate_KG(P_, H_, R_):
    first = P_.dot(H_.T)
    second = H_.dot(P_).dot(H_.T) + R_
    third = np.reciprocal(second, where = second != 0)
    K_ = first.dot(third)
    return K_

# STATE UPDATE EQUATION
def state_update(z_, H_, X_, K_):
    first = z_ - H_.dot(X_)
    second = K_.dot(first)
    X_updated = X_ + second
    return X_updated

# COVARIANCE UPDATE EQUATION
def covariance_update(K_, H_, R_, P_):
    first = np.eye(6) - K_.dot(H_)
    second = first.T
    third = K_.dot(R_).dot(K_.T)
    P_updated = first.dot(P_).dot(second) + third
    return P_updated

def run_kalman(num_steps):
    # Array to store time values
    time_array = np.empty(num_steps)
    x_axis = np.arange(num_steps)

    # Create blank arrays, size of n
    x_walk = np.zeros(num_steps, dtype = np.float64)
    y_walk = np.zeros(num_steps, dtype = np.float64)

    # Create blank arrays, size of 1 -- (n-1) will be appended to each
    x_predict = np.array([0], dtype = np.float64)
    y_predict = np.array([0], dtype = np.float64)

    for i in range(1, num_steps):
        temp = random.randint(1, 4)
        
        if temp == 1:
            x_walk[i] = x_walk[i - 1] + 1
            y_walk[i] = y_walk[i - 1]
        elif temp == 2:
            x_walk[i] = x_walk[i - 1] - 1
            y_walk[i] = y_walk[i - 1]
        elif temp == 3:
            x_walk[i] = x_walk[i - 1]
            y_walk[i] = y_walk[i - 1] + 1
        else:
            x_walk[i] = x_walk[i - 1]
            y_walk[i] = y_walk[i - 1] - 1

    #  Combine Arrays
    walk_data = np.c_[x_walk, y_walk]

    # Initial Conditions
    t = 1   # time between successive measurements (bin size)
    sigma_a = 0.05   # random variance in acceleration
    px, pdx, pddx, py, pdy, pddy = 10, 10, 10, 10, 10, 10   # variances -- since initial state vector is a guess, these values should be high
    sigma_xm, sigma_ym = 3, 3   # measurement error standard deviation (are equal)


    # Initial measurement Vector
    z_n = np.array([[0],
                    [0]])

    # Initial state vector
    X = np.array([[0],
                [0],
                [0],
                [0],
                [0],
                [0]])

    # Initial state transition matrix
    F = np.array([[1, t, 0.5 * t ** 2, 0, 0, 0],
                [0, 1, t, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, t, 0.5 * t ** 2],
                [0, 0, 0, 0, 1, t],
                [0, 0, 0, 0, 0, 1]])

    # Estimate covariance matrix
    P = np.array([[px, 0, 0, 0, 0, 0],
                [0, pdx, 0, 0, 0, 0],
                [0, 0, pddx, 0, 0, 0],
                [0, 0, 0, py, 0, 0],
                [0, 0, 0, 0, pdy, 0],
                [0, 0, 0, 0, 0, pddy]])

    # Observation matrix
    H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]])

    # Process noise matrix for constant acceleration model
    Q = np.array([[t ** 4 / 4, t ** 3 / 2, t ** 2 / 2, 0, 0, 0],
                [t ** 3 / 2, t ** 2 / 2, t, 0, 0, 0],
                [t ** 2 / 2, t, 1, 0, 0, 0],
                [0, 0, 0, t ** 4 / 4, t ** 3 / 2, t ** 2 / 2],
                [0, 0, 0, t ** 3 / 2, t ** 2 / 2, t],
                [0, 0, 0, t ** 2 / 2, t, 1]]).dot(sigma_a)

    # Measurement uncertainty (assumed constant for simplicity)
    R = np.array([[sigma_xm ** 2, 0],
                [0, sigma_ym ** 2]])

    # Initialization
    P_p = cov_extrapolation(P, Q, F)
    X_p = state_extrapolation(X, F)

    i = 0
    reference = time.time_ns() / (10 ** 9)

    for data in walk_data[1:]:
        # Step 1: Measure
        z_n = np.array([[data[0]],
                        [data[1]]])
        
        # Step 2: Update
        K = calculate_KG(P_p, H, R)             # Kalman Gain
        X_c = state_update(z_n, H, X_p, K)      # State
        P_c = covariance_update(K, H, R, P_p)   # Estimate uncertainty (covariance)
        
        # Step 3: Predict
        X_p = state_extrapolation(X_c, F)     # State
        P_p = cov_extrapolation(P_c, Q, F)    # Prediction uncertainty (covariance)

        # Record timestamp
        measure = (time.time_ns() / (10 ** 9))
        time_array[i] = measure - reference
        reference = measure
        i += 1

    return np.average(time_array[1:-1])

#%%
# Metrics
n_arr = [10, 100, 1000, 10000, 100000]
iterations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = np.empty(5)

#%%
rep = 0
for n in n_arr:
    output = np.empty(10)
    for k in range(10):
        output[k] = run_kalman(n)
    results[rep] = np.average(output)
    rep += 1
    

#%%
# Create plot
fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(1, 1, 1)

# Autogenerate graph dimensions
xmin, xmax = min(n_arr), max(n_arr)
ymin, ymax = min(results), max(results)
xint = (xmax - xmin) / 20
yint = (ymax - ymin) / 20
xborder = xint * 2
yborder = yint * 2

# Axes
ax.set_ylim([ymin - yborder, ymax + yborder])

# Log axes
ax.set_xscale('log')

# Line
line = ax.plot(n_arr, results, lw = 0.5, color = 'blue', aa = True, alpha = 0.8, label = 'Average Loop Times')

# Title, legend
ax.set_title('Kalman Filter Time Stamping', fontsize = 20)
ax.legend(fontsize = 12)
plt.xlabel('Number of Steps (n)')
plt.ylabel('Step Time (10e-5 seconds)')

# Tick marks
ax.set_yticks(np.arange(ymin - yborder, ymax + yborder, yint))

# Grid lines
ax.grid(True, which = 'major', linestyle = '--', color = 'gray', alpha = 0.3)

plt.show()

# %%
