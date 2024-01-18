import numpy as np

def rnchisq(df, lambda_):
    if df < 0 or lambda_ < 0:
        return np.nan

    if lambda_ == 0:
        return 0 if df == 0 else np.random.gamma(df / 2.0, 2.0)

    r = np.random.poisson(lambda_ / 2.)
    result = np.random.chisquare(2. * r) if r > 0 else 0
    if df > 0:
        result += np.random.gamma(df / 2.0, 2.0)
    return result

def cir_sim_vec(m):
    EPS = 1e-12
    delta_time = 1.0 / (m + 1.0)
    u_cirsim = np.linspace(delta_time, 1.0 - delta_time, m)
    tau = 0.5 * np.log((u_cirsim * (1 - EPS)) / (EPS * (1.0 - u_cirsim)))
    tau_delta = tau[1:] - tau[:-1]

    # Parameters of CIR
    a, b, sigma = 2.0, 1.0, 2.0 * np.sqrt(2.0)
    c, ncchisq = 0.0, 0.0

    res = np.zeros(m)
    res[0] = np.random.gamma(0.5, 2.0)

    for i in range(1, m):
        c = 2.0 * a / (sigma**2 * (1.0 - np.exp(-a * tau_delta[i-1])))
        ncchisq = rnchisq(4.0 * a * b / sigma**2, 2.0 * c * res[i-1] * np.exp(-a * tau_delta[i-1]))
        res[i] = ncchisq / (2.0 * c)

    return res

def cir_sim_mat(nsim, nobs,random_state=1):
    np.random.seed(random_state)
    res = np.zeros((nsim, nobs))
    for i in range(nsim):
        res[i, :] = cir_sim_vec(nobs)
    return res

