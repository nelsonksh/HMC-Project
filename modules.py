import numpy as np



def bin_Xpol_cl(inarr):

        outarr=np.zeros(24)
        bins=[2, 5, 8, 12, 16, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]

        for i in range(24):
                outarr[i]=np.mean(inarr[bins[i]:bins[i+1]])

        return outarr




def planck_bnu(nu,Td):

######################################################
# This routine computes the Planck blackbody function
# nu is in GHz and temperature is few K.
######################################################

    c = 299792458.
    h = 6.62607554e-34
    htimes10e20 = 6.62607554e-14
    k = 1.38065812e-23
    hoverk = h/k
    nuGHz=nu*1.e9
    out=2*(htimes10e20*nuGHz)*(nuGHz/c)**2 / (np.exp(hoverk*nuGHz/Td) - 1)
    return out

