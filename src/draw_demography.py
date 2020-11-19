"""Functions to plot population size."""

import numpy as np


class Epoch:
    """Length of time to define population size."""

    def __init__(self, n_i, n_f, beta, t_i, t_f):
        """Initialize the class."""
        self.n_i = n_i
        self.n_f = n_f
        self.beta = beta
        self.t_i = t_i
        self.t_f = t_f

    def __str__(self):
        """Representation of the epoch."""
        if self.beta is None:
            s = "N_i: %f,N_f: %f, t_i: %f, t_f %f" % (
                self.n_i,
                self.n_f,
                self.t_i,
                self.t_f,
            )
        else:
            n_i = self.n_i
            n_f = self.n_f
            beta = self.beta
            t_i = self.t_i
            t_f = self.t_f
            s = "N_i: %f,N_f: %f,Beta: %f,t_i: %f, t_f %f" % (n_i, n_f, beta, t_i, t_f)
        return s


def read_demography(demo_file):
    """Read a demography from a file and creates a list of epochs.

    Arguments:
      * demo_file - file listing demographic epochs line-by-line

    """
    demo = []
    demo_str = ""
    t = 0.0
    with open(demo_file, "r") as f:
        for line in f:
            splt = line.split()
            if splt[0] == "g":
                Ni = float(splt[1])
                Nf = float(splt[2])
                b = float(splt[3])
                T = float(splt[4])
                e = Epoch(Ni, Nf, b, t, t + T)
                t += T
                splt[0] = "-g"
            elif splt[0] == "c":
                N = float(splt[1])
                T = float(splt[2])
                T = 10000.0 if T == np.Inf else T
                e = Epoch(N, N, None, t, t + T)
                t += T
                splt[0] = "-c"
            else:
                raise IOError("Invalid Definition of Demographic Epochs")
            demo.append(e)
            demo_str += " ".join(splt) + " "
    return (demo_str, demo)


def generate_cons(con_epoch):
    """Draws a constant demographic Epoch.

    Arguments:
    * con_epoch - an epoch of constant population size

    """
    t_i = con_epoch.t_i
    t_f = con_epoch.t_f
    N = con_epoch.n_i
    tx = np.array([t_i, t_f])
    ns = np.array([N, N])
    return (tx, ns)


def generate_gen(gen_epoch):
    """Draws a generalized growth demographic epoch.

    Arguments:
    *gen_epoch - generalized growth epoch

    """
    t_i = gen_epoch.t_i
    t_f = gen_epoch.t_f
    bk = gen_epoch.beta
    n_i = gen_epoch.n_i
    n_f = gen_epoch.n_f
    tx = np.arange(start=t_i, stop=t_f, step=0.5)
    nt = []
    rk = 0.0
    if bk == 1.0:
        rk = np.log(n_i) - np.log(n_f)
        rk /= t_f - t_i
    else:
        alpha = 1.0 - bk
        rk = (n_f ** alpha) - (n_i ** alpha)
        rk = rk / ((bk - 1.0) * (t_f - t_i))
    for t in tx:
        if bk == 1.0:
            cur_n = n_i * np.exp(-rk * (t - t_i))
            nt.append(cur_n)
        else:
            cur_n = (n_i ** alpha) - (rk * (t - t_i) * alpha)
            cur_n = cur_n ** (1.0 / alpha)
            nt.append(cur_n)
    return (tx, nt)


def generate_demography(demography):
    """Generate points for an entire demographic history.

    Arguments:
    * demography - list of epochs defining the demography

    """
    t = []
    nt = []
    for e in demography:
        if e.beta is None:
            (tx, ns) = generate_cons(e)
        else:
            (tx, ns) = generate_gen(e)
        t.append(tx)
        nt.append(ns)
    t = np.concatenate(t)
    nt = np.concatenate(nt)
    return (t, nt)
