#!/usr/bin/env python

#---------------------------------------------------------
#---------------------------------------------------------
# PURPOSE:
# Code to produce simulated maps
# EXAMPLE:
# MODIFICATION HISTORY:
# 29-07-22 l updated and clean the code
# 29-08-19 ; initiated
#---------------------------------------------------------
#---------------------------------------------------------


import numpy as np
import healpy as hp
import sys
import astropy.io.fits as pyfits
from modules import planck_bnu


simdir='sim_intensity_maps/'
inputpath='maps/'
cibdir='PIP_CIBXXX/'
maskdir='maskdir/'
tmppath='noise_variance/'


nside=int(sys.argv[1])
rnside=int(sys.argv[2])
freq = int(sys.argv[3])
offset = float(sys.argv[4])
noisety=str(sys.argv[5])
nhi_cut = float(sys.argv[6])

ll=str(nhi_cut)
nhi_label = ll.replace(".", "p")


print(offset, freq)

npix= hp.nside2npix(nside)
rnpix=hp.nside2npix(rnside)
nlmax=2*rnside-1


##################################################
#### downgrading the realistic emissivity maps ###
##################################################

alpha= hp.read_map('input_sim_emissivity/Data_LVC_Emissivity_HMC_f353_ns512_rns32_CMB_SUBTRACTED_PIPXVII.fits',field=None,nest=False)
alpha= hp.ud_grade(alpha,nside_out=rnside,order_in='RING', order_out="NESTED")
alpha= alpha*(freq*1./353.)**(1.5)*planck_bnu(freq,20.)/planck_bnu(353,20.)


II=hp.read_map(tmppath + 'new_II_%s_ns_%d_fwhm_16arcm_R3.00_full_MJy_sr2.fits'%(freq,nside),nest=True)
II=np.sqrt(II)

ell, cl= np.loadtxt(cibdir+'CIB_model_Planck14_XXX_%dGHz.txt' %(freq), usecols=(0,1), unpack=True)
cl=cl*1e-12 #MJySr^-1
cl[0:2]=0.

cib=hp.synfast(cl, nside, lmax=None, mmax=None, pol=False, pixwin=True, fwhm=np.radians(16.2/60.))
cib=hp.reorder(cib,r2n=True)

#if(noisety == 'IIcibHII'):
#        ell=np.arange(3000)
#        if(freq==217): cl_g=3.5e-7*ell**(-2.4)
#        if(freq==353): cl_g=1.0e-5*ell**(-2.4)
#        if(freq==545): cl_g=1.6e-4*ell**(-2.4)
#        if(freq==857): cl_g=1.9e-3*ell**(-2.4)
#        cl_g[0]=0.
#        cl_g[1]=0.
#        residual_g=hp.synfast(cl_g, nside, lmax=None, mmax=None, pol=False, pixwin=True, fwhm=np.radians(16.2/60.))
#        residual_g=hp.reorder(residual_g,r2n=True)


### mask file #####
mask=hp.read_map(maskdir+'mask_PIPXVII.fits',nest=True,field=None)
if(np.size(mask) !=npix):
        mask=hp.ud_grade(mask,nside_out=nside,order_in='NESTED', order_out="NESTED")
        ind=np.where(mask <= 0.9)
        mask[ind]=0.0
        ind=np.where(mask > 0.9)
        mask[ind]=1.0

lvc = hp.read_map(inputpath + 'GASS_512_ring_local.fits',field=None,nest=True)
if(np.size(mask) !=npix): lvc=hp.ud_grade(lvc,nside_out=nside,order_in='NESTED', order_out="NESTED")
pixid=np.where(lvc >= nhi_cut)
mask[pixid]=0.0
hp.write_map(maskdir+"mask_PIPXVII_nhi_%s.fits"%(nhi_label),mask,nest=True,overwrite=True)


win=np.zeros(3*nside)
for i in range(3*nside):
    if(i > 90):
        win[i]=1.0
    elif((i > 70) & (i <= 90)):
        win[i]= 0.5*(1.- np.cos(np.pi*(i-70)/20.))
    else:
        win[i]=0


HIresCl = np.ndarray(shape = (3*nside))
el = np.arange(0, 3*nside, 1)
HIresCl[2:] = (el[2:])**(-2.8)*win[2:]
HIresCl[0] = 0.0
HIresCl[1] = 0.0 #dipole is also kept zero


#########################
## galactic residuals realization maps #####
##########################
map_tmp=hp.synfast(HIresCl, nside, lmax=3*nside-1, pixwin=True, fwhm=np.radians(16.2/60))
map_tmp=hp.reorder(map_tmp,r2n=True)
residual_g =map_tmp*lvc**0.8*np.sqrt(1.e-3/0.0033)

if(freq==217): fac_g=0.0094
if(freq==353): fac_g=0.056
if(freq==545): fac_g=0.25
if(freq==857): fac_g=1.0

indexing=int((nside/rnside)**2)
model=np.zeros(npix)
model_test=np.zeros(npix)
tmp_model=np.zeros(npix)

for i in range (rnpix):
        tem= [t for t in range(i*indexing , (i+1)*indexing) if lvc[t]> 0.]
        tmp_model[tem]=alpha[i]
        model_test[tem]=alpha[i]*lvc[tem]+offset
        if(noisety=='II'): model[tem]=alpha[i]*lvc[tem]+offset+ np.random.normal(loc=0.0, scale=II[tem])
        if(noisety=='10II'): model[tem]=alpha[i]*lvc[tem]+offset+ np.random.normal(loc=0.0, scale=II[tem])+ np.random.normal(loc=0.0, scale=10*II[tem])
        if(noisety=='cib'): model[tem]=alpha[i]*lvc[tem]+offset + cib[tem] 
        if(noisety=='IIcib'): model[tem]=alpha[i]*lvc[tem]+offset + cib[tem] + np.random.normal(loc=0.0, scale=II[tem])
        if(noisety=='IIcibHII'): model[tem]=alpha[i]*lvc[tem]+offset+ np.random.normal(loc=0.0, scale=II[tem]) + cib[tem] + residual_g[tem]*fac_g
        if(i ==rnpix-2): print(np.shape(tem), np.random.normal(loc=0.0, scale=II[tem]))


mask2=np.zeros(npix)
index=[]
ind=[]
avoid_pix=[]
for i in range(rnpix):
        theta,phi= hp.pix2ang(rnside,ipix=i,nest=True,lonlat=False)
        if(theta < 115.*np.pi/180.):
                continue
        tem= [t for t in range(i*indexing , (i+1)*indexing) if mask[t]!=0.]
        if(np.size(tem)*1./(indexing*1.) < 1./3.):
                avoid_pix.append(i)
                continue
        mask2[tem]=1.0
        ind.append(tem)
        index.append(i)


model=model*mask2
model_test=model_test*mask2

print("Toy intensity has Gaussian noise")	

hp.write_map( "%s/ext_mask_f%s_ns%d_rns%d_16p2arcm.fits"%(simdir,freq,nside,rnside),mask2,nest=True,overwrite=True)
hp.write_map( "%s/sim_dust_f%d_ns%d_rns%d_%s_16p2arcm.fits"%(simdir,freq,nside,rnside,noisety),model,nest=True,overwrite=True)
hp.write_map( "%s/model_dust_f%d_ns%d_rns%d_16p2arcm.fits"%(simdir,freq,nside,rnside),model_test,nest=True,overwrite=True)




