import os
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import healpy as hp
import time
import sys
import h5py
import copy


nside  = int(sys.argv[1])
rnside = int(sys.argv[2])
freq   = int(sys.argv[3])
noisety = str(sys.argv[4])
dataty = str(sys.argv[5])
nhi_cut = float(sys.argv[6])


ll=str(nhi_cut)
nhi_label = ll.replace(".", "p")

npix = hp.nside2npix(nside)
rnpix=hp.nside2npix(rnside)
indexing=int((nside/rnside)**2)

cibdir='PIP_CIBXXX/'
maskdir='maskdir/'
inputpath='noise_variance/'

star='CIB_HII_variance/nhi_%s' %(nhi_label)
try:
    os.makedirs(star)
except OSError:
    print ("Directory %s already exists" %star)

npix=hp.nside2npix(nside)
rnpix=hp.nside2npix(rnside)
indexing=int((nside/rnside)**2)

### mask file #####
mask=hp.read_map(maskdir+'mask_PIPXVII_nhi_%s.fits' %(nhi_label),nest=True,field=None)

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



ell, cl= np.loadtxt(cibdir+'CIB_model_Planck14_XXX_%dGHz.txt' %(freq), usecols=(0,1), unpack=True)
cl=cl*1e-12 #MJySr^-1
cl[0:2]=0.

print(cl[0:5])

ell=np.arange(3000)
if(nhi_cut < 9.0):
    if(freq==100): cl_g=3.5e-7*ell**(-2.4)
    if(freq==143): cl_g=3.5e-7*ell**(-2.4)
    if(freq==217): cl_g=3.5e-7*ell**(-2.4)
    if(freq==353): cl_g=1.0e-5*ell**(-2.4)
    if(freq==545): cl_g=1.6e-4*ell**(-2.4)
    if(freq==857): cl_g=1.9e-3*ell**(-2.4)
    cl_g[0]=0.
    cl_g[1]=0.
else:
    stop


if(dataty == 'PR3'):
    II=hp.read_map(inputpath + 'new_II_%d_ns_%d_fwhm_16arcm_R3.00_full_MJy_sr2.fits'%(freq,nside),nest=True)

if(dataty == 'PR4'):
    II=hp.read_map(inputpath + 'npipe_noise_var_map_I_nside%d_beamfwhm16p2arcmin_%dGHz.fits'%(nside,freq),nest=False)
    II=hp.reorder(II, r2n=True)
    if(freq == 217): II = II * 483.7**2
    if(freq == 353): II = II * 287.4**2
    if(freq == 545): II = II * 58.0**2
    if(freq == 857): II = II * 2.3**2


####II=hp.reorder(II,n2r=True)


#########################
## CIB and Galactic residuals covariance matrix #####
##########################

#fwhm in radians
elmax = 2*nside
beam_fwhm=np.radians(16.2/60.)
bl = hp.gauss_beam(fwhm = beam_fwhm, lmax=elmax, pol=False)
pl = hp.pixwin(nside = nside, pol=False,lmax=elmax)


if(noisety =='IIcib' or noisety == 'IIcibHII'):
    cov_cib=[]
    cov_HII=[]
    for i in range(np.size(index)):
        pix_id=ind[i]
        covmat1=np.zeros((np.size(pix_id),np.size(pix_id)))
        if(noisety == 'noise_IIcibHII'): covmat2=np.zeros((np.size(pix_id),np.size(pix_id)))
        for m in range(np.size(pix_id)):
            sum1=np.zeros(np.size(pix_id))
            if(noisety == 'noise_IIcibHII'): sum2=np.zeros(np.size(pix_id))
            sel_id=np.repeat(pix_id[m],np.size(pix_id))
            vec1=hp.pix2vec(nside,sel_id,nest=True)
            vec2=hp.pix2vec(nside,pix_id,nest=True)
            x=vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]
            #print(np.min(x),np.max(x))
            pid1=np.where(x > 1)
            pid2=np.where(x < -1)
            x[pid1]=1.
            x[pid2]=-1.
            Pn=np.zeros((elmax,np.size(pix_id)))
            Pn[0,:]=1.
            Pn[1,:]=x
            for l in range(2,elmax):
                Pn[l,:]=(((2 * l)-1)*x * Pn[l-1,:]-(l-1)*Pn[l-2,:])/float(l)
                sum1= sum1 + (2*l+1)*cl[l]*bl[l]**2*pl[l]**2*Pn[l,:]
                if(noisety == 'IIcibHII'): sum2= sum2 + (2*l+1)*cl_g[l]*bl[l]**2*pl[l]**2*Pn[l,:]
            covmat1[m,:]=sum1/(4.*np.pi)
            if(noisety == 'IIcibHII'): covmat2[m,:]=sum2/(4.*np.pi)
        print(i)
        print(covmat1)
        print(np.size(pid1),np.size(pid2))
        cov_cib.append(covmat1)
        if(noisety == 'IIcibHII'): cov_HII.append(covmat2)


cov_mat=[]

for j in range(np.size(index)):
    if(noisety=='II'): y=np.diagflat(II[ind[j]])
    if(noisety=='10II'): y=101.*np.diagflat(II[ind[j]])
    if(noisety=='IIcib'): y=cov_cib[j]+np.diagflat(II[ind[j]])
    if(noisety=='IIcibHII'): y=cov_cib[j]+np.diagflat(II[ind[j]])+cov_HII[j]
    cov_mat.append(y)

cov_mat = np.array(cov_mat)

filename1 = 'CIB_HII_variance/nhi_%s/cov_%s_f%d_ns%d_rns_%d_%s_nhi_%s_fwhm_16p2arcm_MJy_sr2.h5'%(nhi_label,dataty,freq,nside,rnside,noisety,nhi_label)
with h5py.File(filename1, 'w') as hf:
        for n, d in enumerate(cov_mat):
                hf.create_dataset(name = 'dataset{:d}'.format(n), data=d)








 
