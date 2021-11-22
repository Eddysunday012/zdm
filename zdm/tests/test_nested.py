""" Test nested sampling on simple power-law distribution """

from numpy.ma.core import log
import pandas
import numpy as np
import mpmath
from matplotlib import pyplot as plt

import dynesty
from dynesty import plotting as dyplot

from IPython import embed

# Generate a Faux sample
def build_faux_sample(gamma=-2., NFRB=100, lEmax=50., lEth=40., seed=42):
    Emax = 10**lEmax

    norm = (Emax**(gamma+1) - Eth**(gamma+1))/(1+gamma)

    np.random.seed(seed)
    randu = np.random.uniform(size=NFRB)
    randE = (randu*(gamma+1)*norm + 10**(lEth*(gamma+1)))**(1/(1+gamma))

    # Return
    return randE

lEth = 40.
Eth = 10**lEth
Eval = build_faux_sample(lEth=lEth)
mxEval = np.max(Eval)

def run_powerlaw():

    def loglike_powerlaw(x):
        # Unpack for convenience
        lC, gamma, lEmax = x
        C = 10**lC
        Emax = 10**lEmax

        if Emax < mxEval:
            return -9e9
        #
        misses_term = -C * (Emax**(gamma+1) - Eth**(gamma+1)) / (1+gamma)
        # Hits
        NFRB = len(Eval)
        fterm = NFRB * np.log(C)
        sterm = gamma * np.sum(np.log(Eval))
        hits_term = fterm + sterm

        return misses_term + hits_term

    def prior_transform_powerlaw(utheta):
        ulC, ugamma, ulEmax = utheta
        # Transform
        lC = 40. + 4*ulC
        gamma = -1.1 - 1.9*ugamma
        lEmax = 40.5 + 6*ulEmax

        # Return
        return lC, gamma, lEmax


    '''
    lEmax = np.linspace(40.1, 45., 100)
    LLs = []
    for ilEmax in lEmax:
        LLs.append(loglike_powerlaw((42., -2., ilEmax)))

    plt.clf()
    ax = plt.gca()
    ax.plot(lEmax, LLs)
    plt.show()

    embed(header='59 of test_nested')
    '''

    dsampler = dynesty.DynamicNestedSampler(loglike_powerlaw, 
                                            prior_transform_powerlaw, 
                                            ndim=3,
                                            bound='single', 
                                            sample='rwalk')
    # Run it - takes a few minutes
    dsampler.run_nested()
    dres = dsampler.results                                    

    # Plots
    truths = [42, -2., 43]
    labels = [r'$C$', r'$\gamma$', r'$\log_{10} E_{\rm max}$']

    # Traces
    fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                                fig=plt.subplots(3, 2, figsize=(16, 12)))
    fig.tight_layout()
    plt.savefig('power_law_traces.png')
    print("Wrote traces figure to disk ")

    # Corner
    fig, axes = dyplot.cornerplot(dres, truths=truths, show_titles=True, 
                                title_kwargs={'y': 1.04}, labels=labels,
                                fig=plt.subplots(3, 3, figsize=(15, 15)))
    plt.savefig('power_law_corner.png')
    print("Wrote corner figure to disk ")

    print(f"max Eval = {np.max(Eval)}")

    embed(header='59 of test_nested')

def run_gamma():

    def loglike_gamma(x):
        # Unpack for convenience
        lC, gamma, lEmax = x
        C = 10**lC
        Emax = 10**lEmax

        # Misses
        norm = float(mpmath.gammainc(gamma+1, a=Eth/Emax))
        misses_term =  -(C/Emax) * norm

        # Hits
        NFRB = len(Eval)
        fterm = NFRB * (np.log(C) - 2*np.log(Emax))
        sterm= np.sum(np.log((Eval/Emax)**(gamma) * np.exp(-Eval/Emax)))
        hits_term = fterm + sterm

        return misses_term + hits_term

    def prior_transform_gamma(utheta):
        ulC, ugamma, ulEmax = utheta
        # Transform
        lC = 40. + 4*ulC
        gamma = -1.1 - 1.9*ugamma
        lEmax = 40.5 + 6*ulEmax

        # Return
        return lC, gamma, lEmax


    # dyensty
    dsampler = dynesty.DynamicNestedSampler(loglike_gamma, 
                                            prior_transform_gamma, 
                                            ndim=3,
                                            bound='single', 
                                            sample='rwalk')
    # Run it - takes a few minutes
    dsampler.run_nested()
    dres = dsampler.results                                    

    # Plots
    truths = [42, -2., 43]
    labels = [r'$C$', r'$\gamma$', r'$\log_{10} E_{\rm max}$']

    # Traces
    fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                                fig=plt.subplots(3, 2, figsize=(16, 12)))
    fig.tight_layout()
    plt.savefig('gamma_traces.png')
    print("Wrote traces figure to disk ")

    # Corner
    fig, axes = dyplot.cornerplot(dres, truths=truths, show_titles=True, 
                                title_kwargs={'y': 1.04}, labels=labels,
                                fig=plt.subplots(3, 3, figsize=(15, 15)))
    plt.savefig('gamma_corner.png')
    print("Wrote corner figure to disk ")

    print(f"max Eval = {np.max(Eval)}")

    embed(header='59 of test_nested')


# Run
#run_powerlaw()
run_gamma()