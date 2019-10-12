import numpy as np

def geometric_brownian(S0,N,mu,sigma):
    dt = 1 # time interval
    # T = 22 # time horizon for the model
    # t = np.arange(1, int(N) + 1) # saves the predicted values of the model
    n = round(N / dt)
    t = np.linspace(0, N, n)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    return S


y = []
mu = 0.001
sigma = 0.002
