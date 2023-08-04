# Scripts used in simple tests

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from .piDMD import piDMD


def plot_matrices(trueA, piA_data, exA_data):
    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(16, 10)
    titles = ["True A", "pi DMD", "exact DMD",
              "exDMD - piDMD", "truth - piDMD", "truth - exDMD"]
    data = [trueA, piA_data, exA_data,
            exA_data - piA_data, trueA - piA_data, trueA - exA_data]
    for title, ax, dat in zip(titles, axes.flatten(), data):
        heatmap = sns.heatmap(dat.real, ax=ax)
        heatmap.set_title(title)

def plot_eigenvalues(trueVals, piVals, exVals, pi_title="piDMD"):
    plt.figure(figsize=(8, 8))
    circle_data = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
    plt.plot(circle_data.real, circle_data.imag, '--', linewidth=1)
    plt.plot(exVals.real, exVals.imag, 'r^', label="exact DMD")
    plt.plot(piVals.real, piVals.imag, 'bx', label=pi_title, markersize=10)
    plt.plot(trueVals.real, trueVals.imag, 'o', label="truth")
    
    plt.legend()
    plt.title("Spectrum of linear operator")
    plt.ylabel("Im")
    plt.xlabel("Re")
    plt.grid()
    plt.show()
    
def simple_test(
    n: int = 10,
    m: int = 1000,
    trueA: np.ndarray = np.eye(10),
    method: str = "exact",
    rank: int = None,
    diag: np.ndarray = None,
    block_sizes: np.ndarray = None,
    noiseMag: int = 0.5
):
    """
    Inputs:
        n - number of features
        m - number of samples
        trueA - ground truth for matrix A
        method, rank, diag, block_sizes - piDMD parameters
        noiseMag - magnitude of noise added to X and Y before applying DMDs
    """
    
    # generate random but consistent data
    X = np.random.randn(n, m)
    Y = trueA @ X
    
    # make the data noisy
    Y_rand = np.random.randn(*Y.shape)
    X_rand = np.random.randn(*X.shape)
    Yn = Y + noiseMag * Y_rand
    Xn = X + noiseMag * X_rand
    
    # train the models
    ## Physics-Informed DMD
    piA, piVals, piVecs = piDMD(Xn, Yn, method=method, rank=rank,
                                diag=diag, block_sizes=block_sizes)
    exA, exVals, exVecs = piDMD(Xn, Yn, "exact")
    
    # display the error between the learned operators
    I = np.eye(n)
    norm_type = "fro"
    try:
        print(
            f"piDMD {method} model error is {LA.norm(piA(I) - trueA, ord=norm_type) / LA.norm(trueA, ord=norm_type)}"
        )
    except:
        print(
            f"piDMD {method} model error is {LA.norm(piA - trueA, ord=norm_type) / LA.norm(trueA, ord=norm_type)}"
        )
    print(
        f"exact DMD model error is {LA.norm(exA(I) - trueA, ord=norm_type) / LA.norm(trueA, ord=norm_type)}"
    )
    
    return piA, piVals, piVecs, exA, exVals, exVecs