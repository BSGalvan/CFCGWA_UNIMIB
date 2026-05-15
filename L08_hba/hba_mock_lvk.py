import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import numpy as np
import matplotlib.pyplot as plt


from scipy.interpolate import InterpolatedUnivariateSpline as spline


from eryn.ensemble import EnsembleSampler
from eryn.state import State
# from eryn.prior import PriorContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, DistributionGenerateRJ
from eryn.moves.tempering import make_ladder
from scipy.interpolate import RegularGridInterpolator as splinend
from eryn.prior import ProbDistContainer, uniform_dist

import pdfs as pdfs_func

from astropy.cosmology import Planck15
import astropy.units as u

# import corner as triangle



np.random.seed(1)


def compute_selection(params_gauss,params_pl,nrep):

    pdfs_m1_grid=pdfs_func.m1_pdf(m1_grid[None,:],params_gauss,params_pl,mlow=mlow*np.ones(nrep)[:,None],mhigh=mhigh*np.ones(nrep)[:,None])

    # breakpoint()

    ans=np.trapz(pdfs_m1_grid*pdets_m1,x=m1_grid,axis=1)

    return ans


def loglike(xs):

   
    xgauss,xpl = xs

    nrep=len(xgauss)
   
    params_pl={}
    for kpar,par in enumerate(params_vary['pl']):
        params_pl[par]=xpl[:,kpar][:,None]

    params_gauss={}
    for kpar,par in enumerate(params_vary['gauss']):
        params_gauss[par]=xgauss[:,kpar][:,None]

    params_gauss['ampl']=10**params_gauss['log10_ampl_gauss']
    params_pl['ampl']=10**params_pl['log10_ampl_pl']
   
   
    pdf_m1=pdfs_func.m1_pdf(data['m1'][None,:],params_gauss,params_pl,mlow=mlow*np.ones(nrep)[:,None],mhigh=mhigh*np.ones(nrep)[:,None])
    pdf=pdf_m1
    
    breakpoint()
    
    selection_function_term=compute_selection(params_gauss,params_pl,nrep)
 

    #more sanity checks, in case

    logl[np.isinf(first_logl)]=-1E300

    logl[np.isinf(logl)]=-1E300

    logl[selection_function_term==0]=-1E300

    return logl




#read samples and injections

mlow=2
mhigh=100

file_data=np.load('data_mock_lvk.npz')

nevents=np.shape(file_data['m1s_samples'])[0]
nsamp=np.shape(file_data['m1s_samples'])[1]

data={}
data['m1']=file_data['m1s_samples'].reshape(-1)
data['priors']=file_data['priors'].reshape(-1)

m1_grid=np.linspace(mlow,mhigh,1000)
pdets_m1=pdfs_func.pdet(m1_grid)

    
#define branches and parameters of each branch

branch_names=['gauss','pl']

params_vary={}
params_vary['pl']=['log10_ampl_pl','alpha','mmin','mmax']
params_vary['gauss']=['log10_ampl_gauss','mu','sigma']


dict_dims={}
dict_nleaves_min={}
dict_nleaves_max={}

#define number of leaves (components) for each branch 

dict_nleaves_min['pl'],dict_nleaves_max['pl']=1,1
dict_nleaves_min['gauss'],dict_nleaves_max['gauss']=1,1


for branch in branch_names:
    dict_dims[branch]=len(params_vary[branch])

ndims=[]
nleaves_min=[]
nleaves_max=[]
for branch in branch_names:
    ndims+=[dict_dims[branch]]
    nleaves_min+=[dict_nleaves_min[branch]]
    nleaves_max+=[dict_nleaves_max[branch]]


mins={}
maxs={}

mins['mu'],maxs['mu']=20,100
mins['sigma'],maxs['sigma']=1,10
    
mins['log10_ampl_gauss'],maxs['log10_ampl_gauss']=-1.,5.


mins['alpha'],maxs['alpha']=-5,0.
mins['mmin'],maxs['mmin']=2,20
mins['mmax'],maxs['mmax']=30,100

mins['log10_ampl_pl'],maxs['log10_ampl_pl']=-1.,5.



priors={}
for branch in branch_names:
    dict_prior_branch={}
    for kpar,par in enumerate(params_vary[branch]):
        dict_prior_branch[kpar]=uniform_dist(mins[par],maxs[par])
    priors[branch]=ProbDistContainer(dict_prior_branch)


#define sampler settings
nwalkers=20
ntemps=3
nsteps=2000
burn=0
thin_by=1
tempering_kwargs=dict(ntemps=ntemps)
betas = np.linspace(1.0, 0.0, ntemps)


path='run_mock_lvk'

if not os.path.exists(path):
    os.mkdir(path)

path+='/run_%d'%(1)

print(path)

if not os.path.exists(path):
    os.mkdir(path)


#initialise chains

coords = {
    name: priors[name].rvs(size=(ntemps, nwalkers, nleaf,))
    for nleaf, name in zip(nleaves_max, branch_names)
}



state = State(coords)


stretch_move = StretchMove(live_dangerously=True)
moves=stretch_move


# breakpoint()
ensemble = EnsembleSampler(nwalkers,
    ndims,  # assumes ndim_max
    loglike,
    priors,
    tempering_kwargs=dict(betas=betas),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    nleaves_min=nleaves_min,
    provide_groups=False,
    moves=moves,
    vectorize=True,
    rj_moves=False,  # basic generation of new leaves from the prior
    backend=None,
)
# breakpoint()

ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)


samples_gauss = ensemble.get_chain()['gauss'][:,0].reshape(nsteps, nwalkers,dict_dims['gauss'])


samples_pl = ensemble.get_chain()['pl'][:,0].reshape(nsteps, nwalkers, dict_dims['pl'])


loglikes=ensemble.get_log_like()[:,0].reshape(nsteps,nwalkers,1)


breakpoint()

print(np.amax(loglikes))
np.savez(path+'/samples.npz',samples_gauss=samples_gauss,samples_pl=samples_pl,loglikes=loglikes)


# file_config.close()
