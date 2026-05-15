import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
import scipy.special as special



def pdet(x):
    ans= np.exp(-0.5*(x-100)**2/25**2)
    return ans


def m1_pdf(m1,params_gauss,params_pl,norm=False,mlow=2,mhigh=100):

    gauss_part=gaussian_truncated(m1, params_gauss['mu'], params_gauss['sigma'],mlow,mhigh)

    pl_part=power_law(m1,params_pl['alpha'],params_pl['mmin'],params_pl['mmax'])


    ans=params_gauss['ampl']*gauss_part+params_pl['ampl']*pl_part

    if norm:
        ans/=(params_gauss['ampl']+params_pl['ampl'])

    
    return ans


def power_law(x,index,xmin,xmax):

    
    norm=(1/(1+index))*(xmax**(1+index)-xmin**(1+index))

    ans=x**index/norm

    ans[x<xmin]=0
    ans[x>xmax]=0

   

    return ans



def gaussian(x, mean, std):

    return (1/(np.sqrt(2.*np.pi)*std))*np.exp(-((x - mean) ** 2) / (2 * std** 2))


def gaussian_truncated(x, mean, std, mlow, mhigh):

    gauss=gaussian(x, mean, std)

    fact=0.5*(special.erf((mhigh-mean)/(np.sqrt(2)*std))+special.erf((mean-mlow)/(np.sqrt(2)*std)))

    ans=gauss/fact

    try:
        ans[x<mlow]=0

        ans[x>mhigh]=0

    except:
        breakpoint()
    ans[gauss==0]=0

    return ans



def draw_m1(params_gauss,params_pl):

    ms=np.linspace(2,100,1000)
    pdfs=m1_pdf(ms,params_gauss,params_pl,norm=False,mlow=2,mhigh=100)[0]
    pdf_max=np.amax(pdfs)

    draw=True

    while draw:
        m1try=np.random.uniform(2,100)
        pdf_try=m1_pdf(np.array([m1try]),params_gauss,params_pl)[0]
        pkeep=np.random.rand()
        if pkeep*pdf_max<pdf_try:
            draw=False

    return m1try