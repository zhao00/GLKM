import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt

def plot_bar(Y, labels, fname, barWidth=0.3, ra=0.5, addy=10):
    plt.figure(figsize=(6.4, 4.8))

    d_external = (1-2*barWidth)/(1+ra)
    d_internal = ra*d_external

    # Set position of bar on X axis
    X = np.zeros_like(Y)
    x_base = np.arange(Y.shape[0])
    X[:, 0] = x_base - 0.5*d_internal - 0.5*barWidth
    X[:, 1] = x_base + 0.5*d_internal + 0.5*barWidth

    # Make the plot
    plt.bar(X[:, 0], Y[:, 0], color=[0.99, 0.5, 0.05], width=barWidth,
            edgecolor='white', label='Random')
    plt.bar(X[:, 1], Y[:, 1], color=[0.137, 0.533, 0.8], width=barWidth,
            edgecolor='white', label=r'$k$-means')

    for g in range(Y.shape[1]):
        for i in range(Y.shape[0]):
            plt.text(X[i, g], Y[i, g], str(Y[i, g]), rotation=0, horizontalalignment='center')

    plt.xticks(x_base, labels=labels)

    plt.ylim(np.minimum(np.min(Y), 0), np.max(Y) + addy)
    plt.legend(loc='upper left')
    plt.yticks([])
    plt.subplots_adjust(top=0.98, bottom=0.06, right=0.98, left=0.02, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)

    plt.savefig(fname=fname, format="pdf")
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("/home/pei/Hm.C.pdf", dpi=300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

def scatter(X, y, dpi=300, fname=None, show=True, makersize=16):
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=makersize)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    if show:
        plt.show()

def plot(y, dpi=300, xlabel=None, ylabel=None, fname=None, show=True):
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(len(y))
    plt.plot(x, y, "o-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks([])
    # plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.13, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    if show:
        plt.show()
