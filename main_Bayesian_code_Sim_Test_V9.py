#!/usr/bin/env python

#---------------------------------------------------------
#---------------------------------------------------------
# PURPOSE:
# Code for HMC method to compute foreground parameters
# EXAMPLE:
# MODIFICATION HISTORY:
# 05-07-2022 ; taken from v7i -> add nhi cut
# 12-03-2020 ; modified to take realization into account
# 26-02-2020 ; initiated
#---------------------------------------------------------
#---------------------------------------------------------

import os
import numpy as np
import healpy as hp
from astropy.io import fits
import sys
import datetime
import h5py
from modules import planck_bnu
from numpy import linalg

nside  = int(sys.argv[1])
rnside = int(sys.argv[2])
freq   = int(sys.argv[3])
Nsteps =  int(sys.argv[4])
beta = float(sys.argv[5])
noisety = str(sys.argv[6])
e=float(sys.argv[7])
e1=float(sys.argv[8])
relz=int(sys.argv[9])
nhi_cut = float(sys.argv[10])


ll=str(nhi_cut)
nhi_label = ll.replace(".", "p")

outputdir='HMC_outputs/sims/nhi_%s/' %(nhi_label)
try:
    os.makedirs(outputdir)
except OSError:
    print ("Directory %s already exists" %outputdir)


indexing=int((nside/rnside)**2)
npix=hp.nside2npix(nside)
rnpix=hp.nside2npix(rnside)

simdir="sim_intensity_maps/"


#######################
### seed emissivity ####
#######################

alpha_c= hp.read_map('input_sim_emissivity/Data_LVC_Emissivity_HMC_f353_ns512_rns32_CMB_SUBTRACTED_PIPXVII.fits',field=None,verbose=None,nest=False)
alpha_c= hp.ud_grade(alpha_c,nside_out=rnside,order_in='RING', order_out="NESTED")
alpha_c= alpha_c*(freq*1./353.)**(1.5)*planck_bnu(freq,20.)/planck_bnu(353,20.)
seed_alpha = alpha_c + np.random.normal(loc=0.0, scale=0.05,size=np.size(alpha_c))



### mask file #####
mask=hp.read_map('maskdir/mask_PIPXVII_nhi_%s.fits' %(nhi_label),nest=True,field=None,verbose=None)

##NHI templates
NHI=np.array(hp.read_map('maps/GASS_%d_ring_local.fits'%(nside),field=None,nest=True,verbose=None))
if(noisety=='IInew'):
        Isim = np.array(hp.read_map(simdir+'sim_dust_f%d_ns%d_rns%d_IIcib_16p2arcm.fits'%(freq,nside,rnside),nest=True,field=None,verbose=None))
elif(noisety=='IIcibnew'):
        Isim = np.array(hp.read_map(simdir+'sim_dust_f%d_ns%d_rns%d_IIcibHII_16p2arcm.fits'%(freq,nside,rnside),nest=True,field=None,verbose=None))
else:
        Isim = np.array(hp.read_map(simdir+'sim_dust_f%d_ns%d_rns%d_%s_16p2arcm.fits'%(freq,nside,rnside,noisety),nest=True,field=None,verbose=None))

const_map=NHI*0.+1.

####NHI=hp.reorder(NHI,n2r=True)
#####Isim=hp.reorder(Isim,n2r=True)


if(noisety == 'IIcib' or noisety=='IIcibHII'):
    hf = h5py.File('CIB_HII_variance/nhi_%s/cov_PR3_f%d_ns%d_rns_%d_IIcib_nhi_%s_fwhm_16p2arcm_MJy_sr2.h5' %(nhi_label,freq,nside,rnside, nhi_label), 'r')
elif(noisety=='IInew'):
    hf = h5py.File('CIB_HII_variance/nhi_%s/cov_PR3_f%d_ns%d_rns_%d_II_nhi_%s_fwhm_16p2arcm_MJy_sr2.h5' %(nhi_label,freq,nside,rnside, nhi_label), 'r')
elif(noisety=='IIcibnew'):
    hf = h5py.File('CIB_HII_variance/nhi_%s/cov_PR3_f%d_ns%d_rns_%d_IIcib_nhi_%s_fwhm_16p2arcm_MJy_sr2.h5' %(nhi_label,freq,nside,rnside,nhi_label), 'r')
else:
    hf = h5py.File('CIB_HII_variance/nhi_%s/cov_PR3_f%d_ns%d_rns_%d_%s_nhi_%s_fwhm_16p2arcm_MJy_sr2.h5' %(nhi_label,freq,nside,rnside,noisety, nhi_label), 'r')


if(noisety == 'IIcibHII'):
    hf2 = h5py.File('HII_variance/nhi_%s/cov_HII_f%d_ns%d_rns_%d_nhi_%s_fwhm_16p2arcm_full_MJy_sr2.h5' %(nhi_label,freq,nside,rnside, nhi_label), 'r')
    
print("Finish reading the covariance matrix")

inv_cov=[]
for i in range(len(hf.keys())):
    n1=hf.get('dataset%d'%(i))
    n1=np.array(n1)
    if(noisety == 'IIcibHII'):
        n3=hf2.get('dataset%d'%(i))
        n3=np.array(n3)
        n1=n1+n3
    n2 = linalg.inv(n1)
    inv_cov.append(n2)




#################
## parameters ##
################
Nf=10


#######################################
### finding the pixels within mask ####
#######################################

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
    ind.append(tem)
    index.append(i)

hdu = fits.PrimaryHDU(avoid_pix)
hdul = fits.HDUList([hdu])
hdul.writeto(outputdir+'avoid_pixels_f%d_ns%d_rns%d_nhi_%s.fits'%(freq, nside,rnside,nhi_label),overwrite=True)


date = datetime.datetime.now()
date_tmp = '%s%s%s' %(date.day, date.month, date.year)

############# ###########
## assigining the mass matrix #######
#################################
m_alpha=np.zeros(np.size(index))
m_beta=0.0

print(np.shape(index),np.shape(ind))

for j in range(np.size(index)):
        m_beta     = m_beta + np.dot(const_map[ind[j]],np.dot(inv_cov[j],const_map[ind[j]]))
        m_alpha[j] = np.dot(NHI[ind[j]],np.dot(inv_cov[j],NHI[ind[j]]))
       
mass_beta=np.zeros(1)
mass_beta[0]=m_beta

np.savetxt(outputdir+'massmat_emissivity_f%d_ns%d_rns%d_%s_nhi_%s.txt'%(freq, nside,rnside,noisety,nhi_label), m_alpha)
np.savetxt(outputdir+'massmat_offset_f%d_ns%d_rns%d_%s_nhi_%s.txt'%(freq, nside,rnside,noisety,nhi_label),mass_beta)
	


#################################
### conjugate momentum #########
##################################
p_alpha = np.zeros((Nsteps, np.size(index)))

############################
##assigning starting point ##
#############################

alpha = seed_alpha[index]

#######################################
#### Assign the random momentum ####
######################################

p_b=np.random.normal(0,np.sqrt(m_beta),Nsteps)
p_beta=p_b
for j in range(np.size(index)):
        p_alpha[:,j]=np.random.normal(0,np.sqrt(m_alpha[j]),Nsteps)
			

#######################################
##### Hamiltonian ####################
########################################

def Hamiltonian(p_alpha,p_beta,p_alpha_star,p_beta_star,alpha,beta,alpha_star,beta_star):

        KE=0
        PE=0
        KE1=0.
        PE1=0.
        chi2=0.

        for j in range(np.size(m_alpha,axis=0)):                        
                KE=KE + p_alpha[j]**2/m_alpha[j]
                KE1=KE1 + p_alpha_star[j]**2/m_alpha[j]
                shi= (Isim[ind[j]]- beta - alpha[j]*NHI[ind[j]])
                shi1= (Isim[ind[j]]- beta_star - alpha_star[j]*NHI[ind[j]])
                PE=PE + np.dot(shi,np.dot(inv_cov[j],shi))
                PE1=PE1 + np.dot(shi1,np.dot(inv_cov[j],shi1))
                chi2=chi2 + (np.dot(shi,np.dot(inv_cov[j],shi))/np.shape(inv_cov[j])[0])


        H=0.5*(KE1+PE1) - 0.5*(KE+PE)
        return(H, chi2)


##########################
### LEAPFROG #############
##########################
def leapfrog(alpha_lp,beta_lp,u_alpha_lp,u_beta_lp,Nf,e,step):

        alpha_lp=np.copy(alpha_lp)
        beta_lp=np.copy(beta_lp)
        u_alpha_lp=np.copy(u_alpha_lp)
        u_beta_lp=np.copy(u_beta_lp)
        Energy=[]

        for N in range(Nf):
                
                ### LP1 ####
                shi_b = 0.
                for j in range(np.size(u_alpha_lp,axis=0)):
                        shi_a = np.dot(inv_cov[j], Isim[ind[j]]-beta_lp-(alpha_lp[j]*NHI[ind[j]]))
                        shi_alpha= np.dot(NHI[ind[j]], shi_a)
                        u_alpha_lp[j] = u_alpha_lp[j] + (e/2.) * shi_alpha
                        shi_b = shi_b + np.sum(shi_a)

                u_beta_lp = u_beta_lp + (e1/2.) * shi_b
                
                ######## LP2  ########  
                alpha_lp = alpha_lp + e * u_alpha_lp/m_alpha
                beta_lp = beta_lp + e1 * u_beta_lp/m_beta

                #### LP3 ###
                shi_b = 0.
                for j in range(np.size(u_alpha_lp,axis=0)):
                        shi_a = np.dot(inv_cov[j], Isim[ind[j]]-beta_lp-(alpha_lp[j]*NHI[ind[j]]))
                        shi_alpha= np.dot(NHI[ind[j]], shi_a)
                        u_alpha_lp[j] = u_alpha_lp[j] + (e/2.) * shi_alpha
                        shi_b = shi_b + np.sum(shi_a)

                u_beta_lp = u_beta_lp + (e1/2.) * shi_b

        return(u_alpha_lp,u_beta_lp,alpha_lp,beta_lp)

#############################
#### MCMC chain ############
###########################


MC_alpha = []
MC_beta  = []
flag=0
for i in range(Nsteps):
        print("step",i)
        if(i==0):
                alpha_0 = alpha
                beta_0  = beta
        u_alpha = p_alpha[i]
        u_beta  = p_beta[i]
        u_alpha_star,u_beta_star,alpha_star,beta_star = leapfrog(alpha_0,beta_0,u_alpha,u_beta,Nf,e,i)
        H,chi2= Hamiltonian(u_alpha,u_beta,u_alpha_star,u_beta_star,alpha_0,beta_0,alpha_star,beta_star)
        select=np.random.uniform(0.,1.)
        if(select < np.exp(-H)):
                alpha_0 = alpha_star
                beta_0  = beta_star
                flag  = flag +1
        print(beta_0, alpha_0)
        print(chi2/np.size(index))
        MC_alpha.append(alpha_0)
        MC_beta.append(beta_0)


MC_beta=np.array(MC_beta)
MC_alpha=np.array(MC_alpha)


################################
## WRITING MCMC chain #########
###############################

hdu_alpha = fits.PrimaryHDU(MC_alpha)
hdu_beta = fits.PrimaryHDU(MC_beta)
    
hdu_alpha.header['freq'] = '%s' %(freq)
hdu_alpha.header['ns_in'] = '%d' %(nside)
hdu_alpha.header['ns_out'] = '%d' %(rnside)
hdu_alpha.header['noisety'] = '%s' %(noisety)
hdu_alpha.header['Nf'] = '%d' %(Nf)
hdu_alpha.header['e'] = '%.2f' %(e)
hdu_beta.header['e1'] = '%.2f' %(e1)
hdu_alpha.header['nhi_cut'] = '%s' %(nhi_label)
date = datetime.datetime.now()
hdu.header['DATE']= '%s-%s-%s' %(date.day, date.month, date.year)


hdu_beta.header['freq'] = '%s' %(freq)
hdu_beta.header['ns_in'] = '%d' %(nside)
hdu_beta.header['ns_out'] = '%d' %(rnside)
hdu_beta.header['noisety'] = '%s' %(noisety)
hdu_beta.header['Nf'] = '%d' %(Nf)
hdu_beta.header['e'] = '%.2f' %(e)
hdu_beta.header['e1'] = '%.2f' %(e1)
hdu_beta.header['nhi_cut'] = '%s' %(nhi_label)
date = datetime.datetime.now()
hdu.header['DATE']= '%s-%s-%s' %(date.day, date.month, date.year)

hdul_alpha = fits.HDUList([hdu_alpha])
hdul_beta = fits.HDUList([hdu_beta])
hdul_alpha.writeto('%sMCMC_emissivity_lp%d_e%.2f_o%.2f_nsteps%d_f%d_ns%d_rns%d_%s_relz%d_nhi_%s.fits'%(outputdir,Nf,e,e1,Nsteps,freq, nside,rnside,noisety,relz,nhi_label),overwrite=True)
hdul_beta.writeto('%sMCMC_offset_lp%d_e%.2f_o%.2f_nsteps%d_f%d_ns%d_rns%d_%s_relz%d_nhi_%s.fits'%(outputdir,Nf,e,e1,Nsteps,freq, nside,rnside,noisety,relz,nhi_label),overwrite=True)

print("Flag", flag)

f = open('%saccepted_chains_lp%d_e%.2f_o%.2f_nsteps%d_f%d_ns%d_rns%d_%s_relz%d_nhi_%s.txt' %(outputdir,Nf,e,e1, Nsteps,freq, nside,rnside,noisety,relz,nhi_label), 'w')
f.write("%i\n" %(flag))
f.close()













