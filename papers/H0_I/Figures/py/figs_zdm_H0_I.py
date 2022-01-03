""" Figures for H0 paper I"""
import os, sys
from typing import IO
import numpy as np
from numpy.lib.function_base import percentile
import scipy
from scipy import stats

import argparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from zdm.craco import loading
from zdm import pcosmic
from zdm import analyze_cube

from IPython import embed

def fig_craco_fiducial(outfile='fig_craco_fiducial.png',
                zmax=2,DMmax=2000,
                norm=2,log=True,
                label='$\\log_{10} \, p(DM_{\\rm EG},z)$',
                project=False,
                conts=False,
                Aconts=[0.01, 0.1, 0.5],
                Macquart=None,title="Plot",
                H0=None,showplot=False):
    """
    Very complicated routine for plotting 2D zdm grids 

    Args:
        zDMgrid ([type]): [description]
        zvals ([type]): [description]
        dmvals ([type]): [description]
        zmax (int, optional): [description]. Defaults to 1.
        DMmax (int, optional): [description]. Defaults to 1000.
        norm (int, optional): [description]. Defaults to 0.
        log (bool, optional): [description]. Defaults to True.
        label (str, optional): [description]. Defaults to '$\log_{10}p(DM_{\rm EG},z)$'.
        project (bool, optional): [description]. Defaults to False.
        conts (bool, optional): [description]. Defaults to False.
        FRBZ ([type], optional): [description]. Defaults to None.
        FRBDM ([type], optional): [description]. Defaults to None.
        Aconts (bool, optional): [description]. Defaults to False.
        Macquart (state, optional): state object.  Used to generat the Maquart relation.
            Defaults to None.
        title (str, optional): [description]. Defaults to "Plot".
        H0 ([type], optional): [description]. Defaults to None.
        showplot (bool, optional): [description]. Defaults to False.
    """
    # Generate the grid
    survey, grid = loading.survey_and_grid(
        survey_name='CRACO_alpha1_Planck18_Gamma',
        NFRB=100, lum_func=1)

    # Unpack
    zDMgrid, zvals, dmvals = grid.rates, grid.zvals, grid.dmvals
    FRBZ=survey.frbs['Z']
    FRBDM=survey.DMEGs
    
    cmx = plt.get_cmap('cubehelix')
    
    ##### imshow of grid #######
    
    # we protect these variables
    zDMgrid=np.copy(zDMgrid)
    zvals=np.copy(zvals)
    dmvals=np.copy(dmvals)
    
    if (project):
        plt.figure(1, figsize=(8, 8))
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        gap=0.02
        woff=width+gap+left
        hoff=height+gap+bottom
        dw=1.-woff-gap
        dh=1.-hoff-gap
        
        delta=1-height-bottom-0.05
        gap=0.11
        rect_2D = [left, bottom, width, height]
        rect_1Dx = [left, hoff, width, dh]
        rect_1Dy = [woff, bottom, dw, height]
        rect_cb = [woff,hoff,dw*0.5,dh]
        ax1=plt.axes(rect_2D)
        axx=plt.axes(rect_1Dx)
        axy=plt.axes(rect_1Dy)
        acb=plt.axes(rect_cb)
        #axx.xaxis.set_major_formatter(NullFormatter())
        #axy.yaxis.set_major_formatter(NullFormatter())
    else:
        plt.figure()
        #rect_2D=[0,0,1,1]
        ax1=plt.axes()
    
    plt.sca(ax1)
    
    plt.xlabel('z')
    plt.ylabel('${\\rm DM}_{\\rm EG}$')
    #plt.title(title+str(H0))
    
    nz,ndm=zDMgrid.shape
    
    
    ixmax=np.where(zvals > zmax)[0]
    if len(ixmax) >0:
        zvals=zvals[:ixmax[0]]
        nz=zvals.size
        zDMgrid=zDMgrid[:ixmax[0],:]
    
    # sets contours according to norm
    if Aconts:
        slist=np.sort(zDMgrid.flatten())
        cslist=np.cumsum(slist)
        cslist /= cslist[-1]
        nAc=len(Aconts)
        alevels=np.zeros([nAc])
        for i,ac in enumerate(Aconts):
            # cslist is the cumulative probability distribution
            # Where cslist > ac determines the integer locations
            #    of all cells exceeding the threshold
            # The first in this list is the first place exceeding
            #    the threshold
            # The value of slist at that point is the
            #    level of the countour to draw
            iwhich=np.where(cslist > ac)[0][0]
            alevels[i]=slist[iwhich]
        
    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray=np.zeros([nc,nz])
        for i in np.arange(nz):
            cdf=np.cumsum(zDMgrid[i,:])
            cdf /= cdf[-1]
            
            for j,c in enumerate(conts):
                less=np.where(cdf < c)[0]
                
                if len(less)==0:
                    carray[j,i]=0.
                    dmc=0.
                    il1=0
                    il2=0
                else:
                    il1=less[-1]
                    
                    if il1 == ndm-1:
                        il1=ndm-2
                    
                    il2=il1+1
                    k1=(cdf[il2]-c)/(cdf[il2]-cdf[il1])
                    dmc=k1*dmvals[il1]+(1.-k1)*dmvals[il2]
                    carray[j,i]=dmc
                
        ddm=dmvals[1]-dmvals[0]
        carray /= ddm # turns this into integer units for plotting
        
    iymax=np.where(dmvals > DMmax)[0]
    if len(iymax)>0:
        dmvals=dmvals[:iymax[0]]
        zDMgrid=zDMgrid[:,:iymax[0]]
        ndm=dmvals.size
    
    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm=dmvals[1]-dmvals[0]
    dz=zvals[1]-zvals[0]
    if norm==1:
        zDMgrid /= ddm
        if Aconts:
            alevels /= ddm
    if norm==2:
        xnorm=np.sum(zDMgrid)
        zDMgrid /= xnorm
        if Aconts:
            alevels /= xnorm
    
    if log:
        # checks against zeros for a log-plot
        orig=np.copy(zDMgrid)
        zDMgrid=zDMgrid.reshape(zDMgrid.size)
        setzero=np.where(zDMgrid==0.)
        zDMgrid=np.log10(zDMgrid)
        zDMgrid[setzero]=-100
        zDMgrid=zDMgrid.reshape(nz,ndm)
        if Aconts:
            alevels=np.log10(alevels)
    else:
        orig=zDMgrid
    
    # gets a square plot
    aspect=nz/float(ndm)
    
    # sets the x and y tics	
    xtvals=np.arange(zvals.size)
    everx=int(zvals.size/5)
    plt.xticks(xtvals[everx-1::everx],zvals[everx-1::everx])

    ytvals=np.arange(dmvals.size)
    every=int(dmvals.size/5)
    plt.yticks(ytvals[every-1::every],dmvals[every-1::every])
    
    im=plt.imshow(zDMgrid.T,cmap=cmx,origin='lower', 
                  interpolation='None',
                  aspect=aspect)
    
    if Aconts:
        styles=['--','-.',':']
        ax=plt.gca()
        cs=ax.contour(zDMgrid.T,levels=alevels,origin='lower',colors="white",linestyles=styles)
        #plt.clim(0,2e-5)
        #ax.clabel(cs, cs.levels, inline=True, fontsize=10,fmt=['0.5','0.1','0.01'])
    ###### gets decent axis labels, down to 1 decimal place #######
    ax=plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i]=labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if '.' in labels[i]:
            labels[i]=labels[i].split('.')[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0
    
    # plots contours i there
    if conts:
        plt.ylim(0,ndm-1)
        for i in np.arange(nc):
            j=int(nc-i-1)
            plt.plot(np.arange(nz),carray[j,:],label=str(conts[j]),color='white')
        l=plt.legend(loc='upper left',fontsize=8)
        #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
                text.set_color("white")

    muDMhost=np.log(10**grid.state.host.lmean)
    sigmaDMhost=np.log(10**grid.state.host.lsigma)
    meanHost = np.exp(muDMhost + sigmaDMhost**2/2.)
    medianHost = np.exp(muDMhost) 
    print(f"Host: mean={meanHost}, median={medianHost}")
    plt.ylim(0,ndm-1)
    plt.xlim(0,nz-1)
    zmax=zvals[-1]
    nz=zvals.size
    #DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
    DM_cosmic = pcosmic.get_mean_DM(zvals, grid.state)
    
    #idea is that 1 point is 1, hence...
    zeval = zvals/dz
    DMEG_mean = (DM_cosmic+meanHost)/ddm
    DMEG_median = (DM_cosmic+medianHost)/ddm
    plt.plot(zeval,DMEG_mean,color='k',linewidth=2,
                label='Macquart relation (mean)')
    plt.plot(zeval,DMEG_median,color='k',
                linewidth=2, ls='--',
                label='Macquart relation (median)')
    l=plt.legend(loc='lower right',fontsize=12)
    #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
    #for text in l.get_texts():
        #	text.set_color("white")
    
    # limit to a reasonable range if logscale
    if log:
        themax=zDMgrid.max()
        themin=int(themax-4)
        themax=int(themax)
        plt.clim(themin,themax)
    
    ##### add FRB host galaxies at some DM/redshift #####
    if FRBZ is not None:
        iDMs=FRBDM/ddm
        iZ=FRBZ/dz
        # Restrict to plot range
        gd = (FRBDM < DMmax) & (FRBZ < zmax)
        plt.plot(iZ[gd],iDMs[gd],'bo',linestyle="")

    # Set limits
    #ax1.set_ylim(0., DMmax)
        
    # do 1-D projected plots
    if project:
        plt.sca(acb)
        cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=20,pad=0.00,cax = acb)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label,fontsize=8)
        
        axy.set_yticklabels([])
        #axy.set_xticklabels([])
        #axx.set_yticklabels([])
        axx.set_xticklabels([])
        yonly=np.sum(orig,axis=0)
        xonly=np.sum(orig,axis=1)
        
        axy.plot(yonly,dmvals) # DM is the vertical axis now
        axx.plot(zvals,xonly)
        
        # if plotting DM only, put this on the axy axis showing DM distribution
        if FRBDM is not None:
            hvals=np.zeros(FRBDM.size)
            for i,DM in enumerate(FRBDM):
                hvals[i]=yonly[np.where(dmvals > DM)[0][0]]
            
            axy.plot(hvals,FRBDM,'ro',linestyle="")
            for tick in axy.yaxis.get_major_ticks():
                        tick.label.set_fontsize(6)
            
        if FRBZ is not None:
            hvals=np.zeros(FRBZ.size)
            for i,Z in enumerate(FRBZ):
                hvals[i]=xonly[np.where(zvals > Z)[0][0]]
            axx.plot(FRBZ,hvals,'ro',linestyle="")
            for tick in axx.xaxis.get_major_ticks():
                        tick.label.set_fontsize(6)
    else:
        cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
        cbar.set_label(label)
        plt.tight_layout()
    
    plt.savefig(outfile)
    plt.close()
    print(f"Wrote: {outfile}")


def fig_craco_varyH0(outfile='fig_craco_varyH0.png',
                zmax=2,DMmax=1500,
                norm=2,log=True,
                label='$\\log_{10} \, p(DM_{\\rm EG},z)$',
                Aconts=[0.05]):
    # Generate the grid
    survey, grid = loading.survey_and_grid(
        survey_name='CRACO_alpha1_Planck18_Gamma',
        NFRB=100, lum_func=1)
    fiducial_Emax = grid.state.energy.lEmax

    plt.figure()
    ax1=plt.axes()

    plt.sca(ax1)
    
    plt.xlabel('z')
    plt.ylabel('${\\rm DM}_{\\rm EG}$')
    #plt.title(title+str(H0))

    # Loop on grids
    for ss, H0, scl, clr in zip(np.arange(4), 
                      [60., 70., 80., 80.],
                      [0., 0., 0., -0.1],
                      ['b', 'k','r', 'gray']):

        # Update grid
        vparams = {}
        vparams['H0'] = H0
        vparams['lEmax'] = fiducial_Emax + scl
        grid.update(vparams)

        # Unpack
        zDMgrid, zvals, dmvals = grid.rates.copy(), grid.zvals.copy(), grid.dmvals.copy()
        FRBZ=survey.frbs['Z']
        FRBDM=survey.DMEGs
        nz,ndm=zDMgrid.shape
    
        ixmax=np.where(zvals > zmax)[0]
        if len(ixmax) >0:
            zvals=zvals[:ixmax[0]]
            nz=zvals.size
            zDMgrid=zDMgrid[:ixmax[0],:]

        slist=np.sort(zDMgrid.flatten())
        cslist=np.cumsum(slist)
        cslist /= cslist[-1]
        nAc=len(Aconts)
        alevels=np.zeros([nAc])
        for i,ac in enumerate(Aconts):
            # cslist is the cumulative probability distribution
            # Where cslist > ac determines the integer locations
            #    of all cells exceeding the threshold
            # The first in this list is the first place exceeding
            #    the threshold
            # The value of slist at that point is the
            #    level of the countour to draw
            iwhich=np.where(cslist > ac)[0][0]
            alevels[i]=slist[iwhich] 

        iymax=np.where(dmvals > DMmax)[0]
        if len(iymax)>0:
            dmvals=dmvals[:iymax[0]]
            zDMgrid=zDMgrid[:,:iymax[0]]
            ndm=dmvals.size
    
        # currently this is "per cell" - now to change to "per DM"
        # normalises the grid by the bin width, i.e. probability per bin, not probability density
        ddm=dmvals[1]-dmvals[0]
        dz=zvals[1]-zvals[0]
        if norm==1:
            zDMgrid /= ddm
            if Aconts:
                alevels /= ddm
        if norm==2:
            xnorm=np.sum(zDMgrid)
            zDMgrid /= xnorm
            if Aconts:
                alevels /= xnorm
        
        # checks against zeros for a log-plot
        orig=np.copy(zDMgrid)
        zDMgrid=zDMgrid.reshape(zDMgrid.size)
        setzero=np.where(zDMgrid==0.)
        zDMgrid=np.log10(zDMgrid)
        zDMgrid[setzero]=-100
        zDMgrid=zDMgrid.reshape(nz,ndm)
        if Aconts:
            alevels=np.log10(alevels)

        
        # gets a square plot
        aspect=nz/float(ndm)
        
        # sets the x and y tics	
        xtvals=np.arange(zvals.size)
        everx=int(zvals.size/5)
        plt.xticks(xtvals[everx-1::everx],zvals[everx-1::everx])

        ytvals=np.arange(dmvals.size)
        every=int(dmvals.size/5)
        plt.yticks(ytvals[every-1::every],dmvals[every-1::every])
        
        #im=plt.imshow(zDMgrid.T,cmap=cmx,origin='lower', 
        #            interpolation='None',
        #            aspect=aspect)
        
        #styles=['--','-.',':']
        ax=plt.gca()
        cs=ax.contour(zDMgrid.T,levels=alevels,
                      origin='lower',colors=[clr],
                      linestyles=['-'])
        # Label
        cs.collections[0].set_label(
            r"$H_0 = $"+f"{H0}, log Emax = {vparams['lEmax']}")
            #plt.clim(0,2e-5)
            #ax.clabel(cs, cs.levels, inline=True, fontsize=10,fmt=['0.5','0.1','0.01'])

        ###### gets decent axis labels, down to 1 decimal place #######
        ax=plt.gca()
        ax.legend(loc='lower right')

        # Ticks
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in np.arange(len(labels)):
            labels[i]=labels[i][0:4]
        ax.set_xticklabels(labels)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        for i in np.arange(len(labels)):
            if '.' in labels[i]:
                labels[i]=labels[i].split('.')[0]
        ax.set_yticklabels(labels)
        ax.yaxis.labelpad = 0
        

    # Finish
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Wrote: {outfile}")


def fig_craco_H0vsEmax(outfile='fig_craco_H0vsEmax.png'):
    # Load the cube
    cube_out = np.load('../Analysis/Cubes/craco_H0_Emax_cube.npz')
    ll = cube_out['ll'] # log10

    # Slurp
    lEmax = cube_out['lEmax']
    H0 = cube_out['H0']
    #
    dE = lEmax[1]-lEmax[0]
    dH = H0[1] - H0[0]
        
    # Normalize
    ll -= ll.max()

    # Plot
    plt.clf()
    ax = plt.gca()

    im=plt.imshow(ll.T,cmap='jet',origin='lower', 
                    interpolation='None', extent=[40.4-dE/2, 43.4+dE/2, 
                                                  60.-dH/2, 80+dH/2],
                aspect='auto', vmin=-4.
                )#aspect=aspect)
    # Color bar
    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(r'$\Delta$ Log10 Likelihood')
    #
    ax.set_xlabel('log Emax')
    ax.set_ylabel('H0 (km/s/Mpc)')
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")


def fig_craco_H0vsF(outfile='fig_craco_H0vsF.png'):
    # Load the cube
    cube_out = np.load('../Analysis/Cubes/craco_H0_F_cube.npz')
    ll = cube_out['ll'] # log10

    # Slurp
    F = cube_out['F']
    H0 = cube_out['H0']
    #
    dF = F[1]-F[0]
    dH = H0[1] - H0[0]
        
    # Normalize
    ll -= ll.max()

    # Plot
    plt.clf()
    ax = plt.gca()

    im=plt.imshow(ll.T,cmap='jet',origin='lower', 
                    interpolation='None', extent=[0.1-dF/2, 0.5+dF/2, 
                                                  60.-dH/2, 80+dH/2],
                aspect='auto', vmin=-4.
                )#aspect=aspect)
    # Color bar
    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(r'$\Delta$ Log10 Likelihood')
    #
    ax.set_xlabel('F')
    ax.set_ylabel('H0 (km/s/Mpc)')
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")

#### ########################## #########################
def main(pargs):

    # Fiducial CRACO
    if pargs.figure == 'fiducial':
        fig_craco_fiducial()

    # Vary H0, Emax
    if pargs.figure == 'varyH0':
        fig_craco_varyH0()

    # H0 vs. Emax
    if pargs.figure == 'H0vsEmax':
        fig_craco_H0vsEmax()

    # H0 vs. F
    if pargs.figure == 'H0vsF':
        fig_craco_H0vsF()


def parse_option():
    """
    This is a function used to parse the arguments for figure making
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("zdm H0 I Figures")
    parser.add_argument("figure", type=str, 
                        help="function to execute: ('fiducial, 'varyH0', 'H0vsEmax')")
    #parser.add_argument('--cmap', type=str, help="Color map")
    #parser.add_argument('--distr', type=str, default='normal',
    #                    help='Distribution to fit [normal, lognorm]')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)


# python py/figs_zdm_H0_I.py varyH0
# python py/figs_zdm_H0_I.py H0vsEmax
# python py/figs_zdm_H0_I.py H0vsF