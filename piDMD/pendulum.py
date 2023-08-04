# Scripts used in test with pendulum for unitary (orthogonal) piDMD
# for conservative physics

import numpy as np
import matplotlib.pyplot as plt


def pendulum(t, y, params):
    """
    Returns y' = f(t, y) where
    y = [theta1, theta1', theta2, theta2'].
    """
    l1, l2, m1, m2, g = params
    
    a = (m1 + m2) * l1
    b = m2 * l2 * np.cos(y[0] - y[2])
    c = m2 * l1 * np.cos(y[0] - y[2])
    d = m2 * l2
    e = -m2 * l2 * y[3] * y[3] * np.sin(y[0] - y[2])\
        -g * (m1 + m2) * np.sin(y[0])
    f = m2 * l1 * y[1] * y[1] * np.sin(y[0] - y[2])\
        -m2 * g * np.sin(y[2])
    
    yp = np.zeros(4)
    yp[0] = y[1]
    yp[2] = y[3]
    yp[1] = (e * d - b * f) / (a * d - c * b)
    yp[3] = (a * f - c * e) / (a * d - c * b)
    
    return yp
    
def plot_data(m_time, measurements, title):
    plt.figure(figsize=(16, 8))
    plt.plot(m_time, measurements[0], label="th1")
    plt.plot(m_time, measurements[1], label="th2")
    plt.plot(m_time, measurements[2], label="th1dt")
    plt.plot(m_time, measurements[3], label="th2dt")
    
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylabel("measurements")
    plt.xlabel("time")
    plt.show()
    
def plot_res(res_time, truth_data, noisy_data, piDMD_res, exDMD_res, var_name, ylbl):
    plt.figure(figsize=(16, 8))
    plt.plot(res_time, truth_data, label="truth")
    plt.plot(res_time, noisy_data, "--", linewidth=0.5, label="noise")
    plt.plot(res_time, piDMD_res, "m--", label="piDMD orth.")
    plt.plot(res_time, exDMD_res, "g--", label="exact DMD")
    
    plt.title("Results for " + var_name)
    plt.legend()
    plt.grid()
    plt.ylabel(ylbl)
    plt.xlabel("time")
    plt.show()