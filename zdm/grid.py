from IPython.terminal.embed import embed
import numpy as np
from zdm import cosmology as cos
from zdm import parameters
from zdm import misc_functions
from zdm import zdm
from zdm import pcosmic
import time

class Grid:
    """A class to hold a grid of z-dm plots
    
    Fundamental assumption: each z point represents FRBs created *at* that redshift
    Interpolation is performed under this assumption.
    
    It also assumes a linear uniform grid.
    """
    
    def __init__(self, survey, state:parameters.State=None):
        """
        Class constructor.

        Args: 
            survey
            state (parameters.State, optional): 
                Defines the parameters of the analysis
                Note, each grid holds the *same* copy so modifying
                it in one place affects them all.
        """
        self.grid=None
        self.survey = survey
        # we need to set these to trivial values to ensure correct future behaviour
        self.beam_b=np.array([1])
        self.beam_o=np.array([1])
        self.b_fractions=None
        # State
        if state is None:
            state = parameters.State()
        self.state = state

        self.source_function=cos.choose_source_evolution_function(
            state.FRBdemo.source_evolution)
        #self.alpha_method=alpha_method
        #
        self.luminosity_function=0
        self.init_luminosity_functions()
    
    def init_luminosity_functions(self):
        if self.luminosity_function==0:
            self.array_cum_lf=zdm.array_cum_power_law
            self.vector_cum_lf=zdm.vector_cum_power_law
            self.array_diff_lf=zdm.array_diff_power_law
            self.vector_diff_lf=zdm.vector_diff_power_law
        else:
            raise ValueError("Luminosity function must be 0, not ",self.luminosity_function)
    
    def pass_grid(self,zDMgrid,zvals,dmvals):
        self.grid=zDMgrid
        self.zvals=zvals
        self.dmvals=dmvals
        #
        self.check_grid()
        self.calc_dV()
        
        # this contains all the values used to generate grids
        # these parameters begin at None, and get filled when
        # ever something is regenerated. They are semi-hierarchical
        # in that if a low-level value is reset, high-level ones
        # get put to None.
    
    
    
    def load_grid(self,gridfile,zfile,dmfile):
        self.grid=zdm.load_data(gridfile)
        self.zvals=zdm.load_data(zfile)
        self.dmvals=zdm.load_data(dmfile)
        self.check_grid()
        self.volume_grid()
    
    
    def check_grid(self):
        
        self.nz=self.zvals.size
        self.ndm=self.dmvals.size
        self.dz=self.zvals[1]-self.zvals[0]
        self.ddm=self.dmvals[1]-self.dmvals[0]
        shape=self.grid.shape
        if shape[0] != self.nz:
            if shape[0] == self.ndm and shape[1] == self.nz:
                print("Transposing grid, looks like first index is DM")
                self.grid=self.grid.transpose
            else:
                raise ValueError("wrong shape of grid for zvals and dm vals")
        else:
            if shape[1] == self.ndm:
                print("Grid successfully initialised")
            else:
                raise ValueError("wrong shape of grid for zvals and dm vals")
        
        #checks that the grid is approximately linear to high precision
        expectation=self.dz*np.arange(0,self.nz)+self.zvals[0]
        diff=self.zvals-expectation
        maxoff=np.max(diff**2)
        if maxoff > 1e-6*self.dz:
            raise ValueError("Maximum non-linearity in z-grid of ",maxoff**0.5,"detected, aborting")
        
        expectation=self.ddm*np.arange(0,self.ndm)+self.dmvals[0]
        diff=self.dmvals-expectation
        maxoff=np.max(diff**2)
        if maxoff > 1e-6*self.ddm:
            raise ValueError("Maximum non-linearity in dm-grid of ",maxoff**0.5,"detected, aborting")
        
        
    def calc_dV(self):
        """ Calculates volume per steradian probed by a survey.
        
        Does this only in the z-dimension (for obvious reasons!)
        """
        if cos.INIT==False:
            print('WARNING: cosmology not yet initiated, using default parameters.')
            cos.init_dist_measures()
        self.dV=cos.dvdtau(self.zvals)*self.dz
        
    
    def EF(self,alpha=0,bandwidth=1e9):
        """Calculates the fluence--energy conversion factors as a function of redshift
        This does NOT account for the central frequency
        """
        if self.state.FRBdemo.alpha_method==0:
            self.FtoE=cos.F_to_E(1,self.zvals,alpha=alpha,bandwidth=bandwidth,Fobs=self.nuObs,Fref=self.nuRef)
        elif self.state.FRBdemo.alpha_method==0:
            self.FtoE=cos.F_to_E(1,self.zvals,alpha=0.,bandwidth=bandwidth)
        else:
            raise ValueError("alpha method must be 0 or 1, not ",self.alpha_method)
    
    def set_evolution(self): #,n,alpha=None):
        """ Scales volumetric rate by SFR """
        #self.sfr1n=n
        #if alpha is not None:
        #    self.alpha=alpha
        #self.sfr=cos.sfr(self.zvals)**n #old hard-coded value
        self.sfr=self.source_function(self.zvals,
                                      self.state.FRBdemo.sfr_n)
        if self.state.FRBdemo.alpha_method==1:
            self.sfr *= (1.+self.zvals)**(-self.state.energy.alpha) #reduces rate with alpha
            # changes absolute normalisation at z=0 according to central frequency
            self.sfr *= (self.nuObs/self.nuRef)**-self.state.energy.alpha #alpha positive, nuObs<nuref, expected rate increases
            
    def calc_pdv(self,beam_b=None,beam_o=None):
        """ Calculates the rate per cell.
        Assumed model: a power-law between Emin and Emax (erg)
                       with slope gamma.
        Efficiencies: list of efficiency response to DM
        So-far: does NOT include time x solid-angle factor
        
        NOW: this includes a solid-angle and beam factor if initialised
        
        This will recalculate beam factors if they are passed, however
        during iteration this is not recalculated
        """
        
        if beam_b is not None:
            self.beam_b=beam_b
            self.beam_o=beam_o
            try:
                x=beam_o.shape
                x=beam_b.shape
            except:
                raise ValueError("Beam values must be numby arrays! Currently ",beam_o,beam_b)
        #self.Emin=Emin
        #self.Emax=Emax
        #self.gamma=gamma
        # linear weighted sum of probabilities: pdVdOmega now. Could also be used to include time factor

        # For convenience and speed up
        Emin = 10**self.state.energy.lEmin
        Emax = 10**self.state.energy.lEmax
        
        # this implementation allows us to access the b-fractions later on
        if (not (self.b_fractions is not None)) or (beam_b is not None):
            self.b_fractions=np.zeros([self.zvals.size,self.dmvals.size,self.beam_b.size])
        
        # for some arbitrary reason, we treat the beamshape slightly differently... no need to keep an intermediate product!
        for i,b in enumerate(self.beam_b):
            for j,w in enumerate(self.eff_weights):
                
                if j==0:
                    self.b_fractions[:,:,i] = self.beam_o[i]*w*self.array_cum_lf(
                        self.thresholds[j,:,:]/b,Emin,Emax,
                        self.state.FRBdemo.gamma)
                else:
                    self.b_fractions[:,:,i] += self.beam_o[i]*w*self.array_cum_lf(
                        self.thresholds[j,:,:]/b,Emin,Emax,
                        self.state.FRBdemo.gamma)
                
        # here, b-fractions are unweighted according to the value of b.
        self.fractions=np.sum(self.b_fractions,axis=2) # sums over b-axis [ we could ignore this step?]
        self.pdv=np.multiply(self.fractions.T,self.dV).T

    def calc_rates(self):
        """ multiplies the rate per cell with the appropriate pdm plot """
        
        
        try:
            self.sfr
        except:
            print("WARNING: no evolutionary weight yet applied")
            exit()
        
        try:
            self.smear_grid
        except:
            print("WARNING: no DMx smearing yet applied")
            exit()
        
        try:
            self.pdv
        except:
            print("WARNING: no volumetric probability pdv yet calculated")
            exit()
        
        self.sfr_smear=np.multiply(self.smear_grid.T,self.sfr).T
            # we do not NEED the following, but it records this info 
            # for faster computation later
            #self.sfr_smear_grid=np.multiply(self.smear_grid.T,self.sfr).T
            #self.pdv_sfr=np.multiply(self.pdv.T,self.sfr)
        
        self.rates=self.pdv*self.sfr_smear
        
        #try:
        #	self.smear_grid
        #except:
        #	print("WARNING: DM grid has not yet been smeared for DMx!")
        #	self.pdv_smear=self.pdv*self.grid
        #else:
        #	self.pdv_smear=self.pdv*self.sfr_smear
        #
        #try:
        #	self.sfr
        #except:
        #	print("WARNING: no evolutionary weight yet applied")
        #else:
        #	self.rates=np.multiply(self.pdv_smear.T,self.sfr).T
            # we do not NEED the following, but it records this info 
            # for faster computation later
            #self.sfr_smear_grid=np.multiply(self.smear_grid.T,self.sfr).T
            #self.pdv_sfr=np.multiply(self.pdv.T,self.sfr)
        
    def calc_thresholds(self, F0, eff_table, bandwidth=1e9, nuObs=1.3e9, nuRef=1.3e9, weights=None):
        """ Sets the effective survey threshold on the zdm grid
        F0: base survey threshold
        eff_table: table of efficiencies corresponding to DM-values
        nu0: survey frequency (affects sensitivity via alpha - only for alpha method)
        nuref: reference frequency we are calculating thresholds at
        """
        # keep the inputs for later use
        self.F0=F0
        self.nuObs=nuObs
        self.nuRef=nuRef
        
        self.bandwidth=bandwidth
        if eff_table.ndim==1: # only a single FRB width: dimensions of NDM
            self.nthresh=1
            self.eff_weights=np.array([1])
            self.eff_table=np.array([eff_table]) # make it an extra dimension
        else: # multiple FRB widths: dimensions nW x NDM
            self.nthresh=eff_table.shape[0]
            if weights is not None:
                if weights.size != self.nthresh:
                    raise ValueError("Dimension of weights must equal first dimension of efficiency table")
            else:
                raise ValueError("For a multidimensional efficiency table, please set relative weights")
            self.eff_weights=weights/np.sum(weights) #normalises this!
            self.eff_table=eff_table
        Eff_thresh=F0/self.eff_table
        
        
        self.EF(self.state.energy.alpha, bandwidth) #sets FtoE values - could have been done *WAY* earlier
        
        self.thresholds=np.zeros([self.nthresh,self.zvals.size,self.dmvals.size])
        # Performs an outer multiplication of conversion from fluence to energy.
        # The FtoE array has one value for each redshift.
        # The effective threshold array has one value for each combination of
        # FRB width (nthresh) and DM.
        # We loop over nthesh and generate a NDM x Nz array for each
        for i in np.arange(self.nthresh):
            self.thresholds[i,:,:]=np.outer(self.FtoE,Eff_thresh[i,:])
        
        
    def smear_dm(self,smear:np.ndarray):#,mean:float,sigma:float):
        """ Smears DM using the supplied array.
        Example use: DMX contribution
        """
        # just easier to have short variables for this
        
        ls=smear.size
        lz,ldm=self.grid.shape
        #self.smear_mean=mean
        #self.smear_sigma=sigma
        self.smear_grid=np.zeros([lz,ldm])
        self.smear=smear
        #for j in np.arange(ls,ldm):
        #	self.smear_grid[:,j]=np.sum(np.multiply(self.grid[:,j-ls:j],smear[::-1]),axis=1)
        #for j in np.arange(ls):
        #	self.smear_grid[:,j]=np.sum(np.multiply(self.grid[:,:j+1],np.flip(smear[:j+1])),axis=1)
        
        # this method is O~7 times faster than the 'brute force' above for large arrays
        for i in np.arange(lz):
            # we need to get the length of mode='same', BUT
            # we do not want it 'centred', hence must make cut on full
            if smear.ndim==1:
                self.smear_grid[i,:] = np.convolve(self.grid[i,:],smear,mode='full')[0:ldm]
            elif smear.ndim==2:
                self.smear_grid[i,:] = np.convolve(self.grid[i,:],smear[i,:],mode='full')[0:ldm]
            else:
                raise ValueError("Wrong number of dimensions for DM smearing ",smear.shape)
            
    def get_p_zgdm(self, DMs):
        """ Calcuates the probability of redshift given a DM
        We already have our grid of observed DM values.
        Just take slices!
        
        """
        # first gets ids of matching DMs
        priors=np.zeros([DMs.size,self.zvals.size])
        for i,dm in enumerate(DMs):
            DM2=np.where(self.dmvals > dm)[0][0]
            DM1=DM2-1
            kDM=(dm-self.dmvals[DM1])/(self.dmvals[DM2]-self.dmvals[DM1])
            priors[i,:]=kDM*self.rates[:,DM2]+(1.-kDM)*self.rates[:,DM1]
            priors[i,:] /= np.sum(priors[i,:])
        return priors
    
    def GenMCSample(self,N,Poisson=False):
        """
        Generate a MC sample of FRB events
        
        If Poisson=True, then interpret N as a Poisson expectation value
        Otherwise, generate precisely N FRBs
        
        Generated values are DM, z, B, w, and SNR
        NOTE: the routine GenMCFRB does not know 'w', merely
            which w bin it generates.
        
        """
        
        if Poisson:
            #from np.random import poisson
            NFRB=np.random.poisson(N)
        else:
            NFRB=int(N) #just to be sure...
        sample=[]
        pwb=None #feeds this back to save time. Lots of time.
        for i in np.arange(NFRB):
            frb,pwb=self.GenMCFRB(pwb)
            sample.append(frb)
        sample=np.array(sample)
        return sample
    
    def GenMCFRB(self,pwb=None):
        """
        Generates a single FRB according to the grid distributions
        
        Samples beam position b, FRB DM, z, s=SNR/SNRth, and w
        Currently: no interpolation included.
        This should be implemented in s,DM, and z.
        
        NOTE: currently, the actual FRB widths are not part of 'grid'
            only the relative probabilities of any given width.
            Hence, this routine only returns the integer of the width bin
            not the width itelf.
        
        """
        # grid of beam values, weights
        nw=self.eff_weights.size
        nb=self.beam_b.size
        
        # we do this to allow efficient recalculation of this when generating many FRBs
        if pwb is not None:
            pwbc=np.cumsum(pwb)
            pwbc/=pwbc[-1]
        else:
            pwb=np.zeros([nw*nb])
            
            # Generates a joint distribution in B,w
            for i,b in enumerate(self.beam_b):
                for j,w in enumerate(self.eff_weights):
                    # each of the following is a 2D array over DM, z which we sum to generate B,w values
                    wb_fraction=self.beam_o[i]*w*self.array_cum_lf(self.thresholds[j,:,:]/b,self.Emin,self.Emax,self.gamma)
                    pdv=np.multiply(wb_fraction.T,self.dV).T
                    rate=pdv*self.sfr_smear
                    pwb[i*nw+j]=np.sum(rate)
            pwbc=np.cumsum(pwb)
            pwbc/=pwbc[-1]
        
        # sample distribution in w,b
        # we do NOT interpolate here - we treat these as qualitative values
        # i.e. as if there was an irregular grid of them
        r=np.random.rand(1)[0]
        which=np.where(pwbc>r)[0][0]
        i=int(which/nw)
        j=which-i*nw
        MCb=self.beam_b[i]
        MCw=self.eff_weights[j]
        
        # calculate zdm distribution for sampled w,b only
        pzDM=self.array_cum_lf(self.thresholds[j,:,:]/MCb,self.Emin,self.Emax,self.gamma)
        wb_fraction=self.array_cum_lf(self.thresholds[j,:,:]/MCb,self.Emin,self.Emax,self.gamma)
        pdv=np.multiply(wb_fraction.T,self.dV).T
        pzDM=pdv*self.sfr_smear
        
        
        # sample distribution in z,DM
        pz=np.sum(pzDM,axis=1)
        pzc=np.cumsum(pz)
        pzc /= pzc[-1]
        r=np.random.rand(1)[0]
        iz2=np.where(pzc>r)[0][0]
        if iz2 > 0:
            iz1=iz2-1
            dr=r-pzc[iz1]
            kz2=dr/(pzc[iz2]-pzc[iz1]) # fraction of way to second value
            kz1=1.-kz2
            MCz=self.zvals[iz1]*kz1+self.zvals[iz2]*kz2
            pDM=pzDM[iz1,:]*kz1 + pzDM[iz2,:]*kz2
        else:
            # we perform a simple linear interpolation in z from 0 to minimum bin
            kz2=r/pzc[iz2]
            kz1=1.-kz2
            MCz=self.zvals[iz2]*kz2
            pDM=pzDM[iz2,:] # just use the value of lowest bin
        
        
        # NOW DO dm
        #pDM=pzDM[k,:]
        pDMc=np.cumsum(pDM)
        pDMc /= pDMc[-1]
        r=np.random.rand(1)[0]
        iDM2=np.where(pDMc>r)[0][0]
        if iDM2 > 0:
            iDM1=iDM2-1
            dDM=r-pDMc[iDM1]
            kDM2=dDM/(pDMc[iDM2] - pDMc[iDM1])
            kDM1=1.-kDM2
            MCDM=self.dmvals[iDM1]*kDM1 + self.dmvals[iDM2]*kDM2
            if iz2>0:
                Eth=self.thresholds[j,iz1,iDM1]*kz1*kDM1 \
                    + self.thresholds[j,iz1,iDM2]*kz1*kDM2 \
                    + self.thresholds[j,iz2,iDM1]*kz2*kDM1 \
                    + self.thresholds[j,iz2,iDM2]*kz2*kDM2
            else: 
                Eth=self.thresholds[j,iz2,iDM1]*kDM1 \
                    + self.thresholds[j,iz2,iDM2]*kDM2
                Eth *= kz2**2 #assume threshold goes as Eth~z^2 in the near Universe
        else:
            # interpolate linearly from 0 to the minimum value
            kDM2=r/pDMc[iDM2]
            MCDM=self.dmvals[iDM2]*kDM2
            if iz2>0: # ignore effect of lowest DM bin on threshold
                Eth=self.thresholds[j,iz1,iDM2]*kz1 \
                    + self.thresholds[j,iz2,iDM2]*kz2
            else: 
                Eth=self.thresholds[j,iz2,iDM2]*kDM2
                Eth *= kz2**2 #assume threshold goes as Eth~z^2 in the near Universe
        
        # now account for beamshape
        Eth /= MCb
        
        # NOW GET snr
        #Eth=self.thresholds[j,k,l]/MCb
        Es=np.logspace(np.log10(Eth),np.log10(self.Emax),100)
        PEs=self.vector_cum_lf(Es,self.Emin,self.Emax,self.gamma)
        PEs /= PEs[0] # normalises: this is now cumulative distribution from 1 to 0
        r=np.random.rand(1)[0]
        iE1=np.where(PEs>r)[0][-1] #returns list starting at 0 and going upwards
        iE2 = iE1+1
        # iE1 should never be the highest energy, since it will always have a probability of 0
        kE1=(r-PEs[iE2])/(PEs[iE1]-PEs[iE2])
        kE2=1.-kE1
        MCE=10**(np.log10(Es[iE1])*kE1 + np.log10(Es[iE2])*kE2)
        MCs=MCE/Eth
        
        FRBparams=[MCz,MCDM,MCb,j,MCs]
        return FRBparams,pwb
        

    def update(self, vparams:dict):
        # Init
        smear_dm, calc_pdv, set_evol, calc_rates = False, False, False, False
        new_sfr_smear, new_pdv_smear, calc_thresh = False, False, False

        # Cosmology -- Only H0 so far
        if self.chk_upd_param('H0', vparams, update=True):
            print("Updating cosmology [H0] for the grid")
            cos.set_cosmology(self.state)
            zDMgrid, zvals,dmvals=misc_functions.get_zdm_grid(
                self.state, new=True,plot=False,method='analytic')
            # TODO -- Check zvals and dmvals haven't changed!
            self.pass_grid(zDMgrid,zvals,dmvals)
            #self.smear_mean=oldsmean
            #self.smear_sigma=oldssigma
            #self.smear_dm(self.smear)#,oldsmean,oldssigma)
            # pass_grid calls calc_dV()
            #self.calc_dV()
            smear_dm = True
            # TODO -- do we need to do this step??  Only if dmvals change
            #self.calc_thresholds(self.survey.meta['THRESH'],
            #                     self.survey.efficiencies,
            #                     weights=self.survey.wplist)
            calc_thresh = True
            calc_pdv = True
            set_evol = True
            calc_rates = True
            #self.calc_pdv(Emin,Emax,gamma,self.survey.beam_b,self.survey.beam_o)
            #self.set_evolution(oldsfrn) 
            #self.calc_rates()

        # Mask?
        # IT IS IMPORTANT TO USE np.any so that each item is executed!!
        if np.any([self.chk_upd_param('lmean', vparams, update=True), 
            self.chk_upd_param('lsigma', vparams, update=True)]):
            self.smear=pcosmic.get_dm_mask(
                self.dmvals,(self.state.host.lmean,
                             self.state.host.lsigma),
                self.zvals)
            smear_dm = True

        # Smear?
        if smear_dm:
            print("grid.py: Smearing dm")
            self.smear_dm(self.smear)

        # SFR?
        if self.chk_upd_param('sfr_n', vparams, update=True):
            set_evol = True
            new_sfr_smear=True  # True for either alpha_method
        if self.chk_upd_param('alpha', vparams, update=True):
            set_evol = True
            calc_pdv = True
            new_pdv_smear=True
            if self.state.FRBdemo.alpha_method == 0:
                calc_thresh = True
            elif self.state.FRBdemo.alpha_method == 1:
                new_sfr_smear=True
        if set_evol:
            self.set_evolution() # sets star-formation rate scaling with z - here, no evoltion...

        # TODO -- Thresholds and pdv come *before* in initialize_grids
        #  SHOULD WE DO THE SAME HERE??

        ##### examines the 'pdv tree' affecting sensitivity #####
        # begin with alpha
        # alpha does not change thresholds under rate scaling, only spec index
        if calc_thresh:
            self.calc_thresholds(
                self.F0,self.eff_table, bandwidth=self.bandwidth,
                weights=self.eff_weights)

        if np.any([self.chk_upd_param('lEmin', vparams, update=True),
            self.chk_upd_param('lEmax', vparams, update=True),
            self.chk_upd_param('gamma', vparams, update=True)]):
            calc_pdv = True
            new_pdv_smear=True
        
        if calc_pdv:
            self.calc_pdv()

        if new_sfr_smear:
            self.calc_rates() #includes sfr smearing factors and pdv mult
        elif new_pdv_smear:
            self.rates=self.pdv*self.sfr_smear #does pdv mult only, 'by hand'

        # Catch all the changes just in case, e.g. lC
        self.state.update_params(vparams)

    def chk_upd_param(self, param:str, vparams:dict, update=False):
        updated = False
        DC = self.state.params[param]
        # In dict?
        if param in vparams.keys():
            # Changed?
            if vparams[param] != getattr(self.state[DC], param):
                updated = True
                if update:
                    self.state.update_param(param, vparams[param])
        #
        return updated

    
    '''
    def copy(self,grid):
        """ copies all values from grid to here
        is OK if this is a shallow copy
        explicit function to open the possibility of making it faster
        This is likely now deprecated, and was made before the difference
        in NFRB and NORM_FRB was implemented to allow two grids with
        identical properties, but only one of which had a normalise time,
        to be rapidly evaluated.
        """
        self.grid=grid.grid
        self.beam_b=grid.beam_b
        self.b_fractions=grid.b_fractions
        self.zvals=grid.zvals
        self.dmvals=grid.dmvals
        self.nz=grid.nz
        self.ndm=grid.ndm
        self.dz=grid.dz
        self.ddm=grid.ddm
        self.dV=grid.dV
        self.FtoE=grid.FtoE
        self.sfr_n=grid.sfr_n
        self.sfr=grid.sfr
        self.beam_b=grid.beam_b
        self.beam_o=grid.beam_o
        self.Emin=grid.Emin
        self.Emax=grid.Emax
        self.gamma=grid.gamma
        self.b_fractions=grid.b_fractions
        self.fractions=grid.fractions
        self.pdv=grid.pdv
        self.smear_grid=grid.smear_grid
        self.sfr_smear=grid.sfr_smear
        self.rates=grid.rates
        self.F0=grid.F0
        self.alpha=grid.alpha
        self.bandwidth=grid.bandwidth
        self.nthresh=grid.nthresh
        self.eff_weights=grid.eff_weights
        self.eff_table=grid.eff_table
        self.thresholds=grid.thresholds
        self.smear_mean=grid.smear_mean
        self.smear_sigma=grid.smear_sigma
        self.smear=grid.smear
        self.smear_grid=grid.smear_grid
        self.nu0=grid.nuObs
        self.nuref=grid.nuRef
    '''