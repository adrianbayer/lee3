# for periodogram can use astropy implementation for fast results. Try some different options for consistency.

import numpy as np
import os, sys
import time
time0 = time.time()
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import argparse

from mpi4py import MPI
from mpi_tools import *

from periodogram_selfcalib_tools import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A', type=float, default=0, help='Amplitude of physical peak')
parser.add_argument('--dk', type=float, default=0, help='For Df modification')
parser.add_argument('--peak_removal_df', type=float, default=0, help='If A>0, then this will remove all peaks in +- of the true f from sc')
args = parser.parse_args()

len_t = 100     # higher len_t makes things break down self-calibration not work as well
T = 1           # T should be arbitrary in terms of what the priodogram looks like
n_batch = 10
squash = 5
fbg = 0         # fixed batch gap
fib = 0         # fixed inner batch

noise_type = 'normal'

A = args.A           # include a real peak in the data?
peak_removal_df = args.peak_removal_df

fminT = 0.5    # choose frequency cut
fmaxTs = len_t/2*np.array([100,10,1])[2:] # [5000, 500, 50][2:]

power_method = 'slow_Df'                       #methods = ['auto', 'slow', 'chi2', 'cython', 'fast', 'fastchi2']
normalization = 'psd'                         # 'psd' is chi2/2  ... 'standard' is (chisq0-chisq)/chisq0

nterms = 1     # Number of Fourier modes in fit
M = 3      
if nterms > 1:
    print("nterms > 1 only works with chi2.")
    power_method = 'chi2'

spp = 50

Nseeds = 1000

dealias = 0

vdPfig1 = False     # use the vdP_fig1 data. I.e. fix t accross realizations, and just chage y(t) each time.

narr = np.array([5,10,20,50,100,200])[:3]#,30])    # list of ns we want to try for self-calibration

#### Df stuff
dk = args.dk
if dk != 0 and 'Df' not in power_method:
    raise Exception("Please set dk=0, or change the power method to one which supports Df.")
# have to do rest of computation inside code after computing t (which can be random)

#####################################################################################

outfolder = "/global/cscratch1/sd/abayer/lookelsewhere/periodogram_astropy/data/"
if n_batch > 1 and squash > 1:
    outfolder += "lent%d_nb%d_s%d_fbg%d_fib%d_fmin%.2e_pm%s_norm%s_spp%d_ns%.0e_da%d_dk%0.2e/" % (len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, dk)
    #fmaxTs = fmaxTs + [(len_t-1)/(2*T*n_batch), (len_t-1)/(2*T)*squash]
else:
    outfolder += "lent%d_nb%d_s%d_fbg%d_fmin%.2e_pm%s_norm%s_spp%d_ns%.0e_da%d_dk%0.2e/" % (len_t, n_batch, squash, fbg, fminT, power_method, normalization, spp, Nseeds, dealias, dk)


if vdPfig1:
    t = np.loadtxt("/global/cscratch1/sd/abayer/lookelsewhere/periodogram_astropy/vdP_fig1.txt", delimiter=',', skiprows=1)[:,0]
    # dodgily normalize to start at 0, and say baseline is the difference between first and last measurement.
    #t = t - t.min()
    T = t.max() - t.min()                 # Need T when asking for fmax in periodogram. fmax = fmaxT/T !
    len_t = len(t)
    print(len_t)
    fmaxTs = [100*(len_t-1)/2, 10*(len_t-1)/2, (len_t-1)/2][1:]     # Careful T is not 1 here, so need to modify code to account for this
    outfolder = "/global/cscratch1/sd/abayer/lookelsewhere/periodogram_astropy/data/vdPfig1_T%d_fmin%.2e_pm%s_norm%s_spp%d_ns%.0e_da%d/" % (T, fminT, power_method, normalization, spp, Nseeds, dealias) 

if nterms > 1:
    outfolder = outfolder[:-1] + '_nterms%d_M%d/' % (nterms, M)
    
if A != 0:
    outfolder = outfolder[:-1] + '_A%.2e/' % A
    if peak_removal_df > 0:
        outfolder = outfolder[:-1] + 'df%.2e/' % peak_removal_df
        
if noise_type != 'normal':
    outfolder = outfolder[:-1] + '_nt%s/' % noise_type
    
####################################################################################
print(fmaxTs)
seeds = np.arange(Nseeds)

# MPI
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# makes array into 0, size, 2*size ...... 1, size + 1, 2*size +1, ....
# this means each proc gets 0, size, 2*size... rather than 0, 1, 2, etc. This helps distribute saving the first few seeds
seeds = seeds.reshape(-1, size).T.flatten()  

if rank == root:
    os.makedirs(outfolder, exist_ok=True)
    print(outfolder)
    
_seeds = scatter_array_by_col(seeds, comm, root)

_QL = np.empty((len(fmaxTs), len(_seeds)))               # local len seeds. will gather at end. using _ for local for neatness
_QS = np.empty((len(narr), len(fmaxTs), len(_seeds)))
_K = np.empty((len(narr), len(fmaxTs), len(_seeds)))

for si,seed in enumerate(_seeds):
    #if size > 1:
    #    if rank == root and si%int(Nseeds/size/50)==0:
    #        print(rank, si, seed, seed/Nseeds, time.time() - time0)
    print(seed)   
    # generate time data
    if not vdPfig1:
        t = generate_times(len_t, T, n_batch, squash, fbg, fib, seed)
        # del half
        #np.random.seed(seed)
        #t = np.random.choice(t, int(len_t/2), replace=False)
    
    # compute Df
    aaa = 2*dk/T**2
    Df = aaa * (t - T/2)
        
    # model params
    ###M = 3                 # for A = 0, phi=0 doens't matter. So really should have M=3 I think. Can just think of periodogram as 2dof chisq => M=3
    f = np.random.RandomState(seed).rand(1) * len_t / T / 2
    phi = 0  # irreleavnt because A = 0. Will keep in code in case I ever want to play around with real signal
    dy = 1
    
    if noise_type == 'normal':
        noise = dy * np.random.RandomState(seed).randn(len_t)
    elif noise_type == 'poisson':
        noise = np.random.RandomState(seed).poisson(dy*5, len_t) * 0.2
        noise = noise - np.mean(noise)
    elif noise_type == 'nct':
        from scipy.stats import nct
        noise = dy * np.random.RandomState(seed).randn(len_t)
        
        mask = np.random.random(len_t) < 0.02 #fraction of all points that are outliers
        noise[mask] = nct.rvs(df = 1.55134374, nc = 0.86912289, loc = 1.98435008, scale = 0.71418219, 
                              size = np.sum(mask), 
                              random_state = np.random.RandomState(seed))
        #print(noise)
    
    y = A * np.sin(2 * np.pi * f * t + phi) + noise
        
    # save the data
    if seed < 100:
        seeddir = os.path.join(outfolder, 'seed'+str(seed).zfill(6))
        os.makedirs(seeddir, exist_ok=True)
        np.savetxt(os.path.join(seeddir, 'data.txt'), np.vstack((t,y)).T)
        
    for fTi,fmaxT in enumerate(fmaxTs):
        
        # recompute y for each fmaxT, becuase for dealiasing we need to refresh y back to what it was). FIXME could copy y0 from outside loop instead
        y = A * np.sin(2 * np.pi * f * t + phi) + noise
        
        # self-calibrate with dealiasing (evn if you dont dealiase, you want to start this loop, and then it gets broken)
        n_sc_i = 0   # n used for self callib
        nmaxiter = narr.max() + 1 #if normalization == 'psd' else int(2*(narr.max() + 1))    # we want to calibrate k with this deeper peak.
        for n in range(0, nmaxiter):
            ls = LombScargle(t, y, dy, normalization=normalization, fit_mean=False, center_data=False, nterms=nterms)
            freq = ls.autofrequency(minimum_frequency=fminT/T, maximum_frequency=fmaxT/T, samples_per_peak=spp)
            #freq = np.linspace(fminT, fmaxT, 1000*int(fmaxT)+1)
            power = ls.power(freq, method=power_method, Df=Df)
            
            # save global maximum, pre-dealiasing. Also break (shortly after) if no dealiasing
            if n == 0:
                power0 = ls.power(freq, method=power_method, Df=0)   # for the maximum qL we always use Df=0 and don't remove any peaks
                qL_max = 2 * power0.max()  
                _QL[fTi, si] = qL_max
            
            # FIXME... not as ideal as the find_peak method for not aliasing, but we want to define qL_max appropriately...
            # right now this removes the peaks from QL completely, even for the qL pval line. This might not be what we want really.
            # prob only want to remove from sc... but for now this is here for intuition.
            if A > 0 and peak_removal_df > 0:
                mask = (freq < f - peak_removal_df) + (freq > f + peak_removal_df)
                freq = freq[mask]
                power = power[mask]
            
            if not dealias:
                break
            
            # self calibrate if we're at an n we want to self calibrate for
            if n in narr:    # might want to do n+1 in narr because 0 v 1 index, but not a huge deal. This is what I've been using elsewhere
                qL_n = 2 * power.max()    # on iteration n (0 index) the maximum will give the peak with n peaks removed from likelihood. so n=1 gives second highest pre-dealiasing
                qs_n = qL_max - qL_n - 2*np.log(n+1) - (M-2) * np.log(qL_max / qL_n)      # using tau = q for simplicity here.
                _QS[n_sc_i, fTi, si] = qs_n
                n_sc_i += 1
                #print(seed, fmaxT, n, qL_max, qL_n, qs_n)
            
            # find best fit f, and remove the mode from the data
            f_fit = freq[np.argmax(power)]
            y_fit = ls.model(t, f_fit)     # this is contribution of the (current) maximum frequency mode to y
            
            #y_fit2 = theta[0] * np.sin(2*np.pi*f_fit*t) + theta[1] * np.cos(2*np.pi*f_fit*t) 
            #print(y_fit - y_fit2)
            
            # save peak loc info at each iteration and also the best fit information
            if seed < 100:
                theta = None
                if n == 0:    # save all the theta information for the non-aliased profile
                    theta = [ls.model_parameters(ff) for ff in freq]    #FIXME, could try vectorize
                theta_fit = ls.model_parameters(f_fit)        # compute theta before modifying y or you get a bug... FIXME: maybe there's a bug in y in terms of y-=? Myabe ls changes? I doubt it
                peak_ids, _ = find_peaks(power)
                np.savez(os.path.join(seeddir, 'peak_locs_info_fmaxT%.2e_nda%d.npz' % (fmaxT, n)), freq=freq, power=power, theta=theta, peak_indices=peak_ids, 
                         f_fit=f_fit, theta_fit=theta_fit)
            
            y -= y_fit                     # remove the best fit from the data. do this after computing theta or you get a bug. Probably copying related.
        
        # self-calibration without dealiasing
        if not dealias:
            # find peaks in 1d data using scipy's find_peaks        
            
            peak_ids, _ = find_peaks(power)
            #if A > 0 and peak_removal_df > 0:
            #    acceptedregion = np.where((freq < f - peak_removal_df) + (freq > f + peak_removal_df))[0]
            #    peak_ids = np.intersect1d(peak_ids, acceptedregion)
                
            peak_qLs = 2 * power[peak_ids]

            peak_argsort = np.argsort(peak_qLs)[::-1]
            peak_qLs = peak_qLs[peak_argsort]

            qL_sorted = peak_qLs

            for ni,nn in enumerate(narr):
                try:
                    if normalization == 'psd':
                        tau_n = (qL_sorted[nn]+qL_sorted[nn+1])/2
                        k = 1
                    else:
                        mm = int(nn/4)    # can tweak this FIXME make input parameter
                        tau_n = ((qL_sorted[nn]+qL_sorted[nn])/2) # qL_sorted[nn]
                        tau_m = ((qL_sorted[mm]+qL_sorted[mm])/2) # qL_sorted[mm]
                        k = ( 2*np.log((nn+1)/(mm+1)) + (M-2)*np.log(tau_m/tau_n) ) / (tau_m - tau_n)
                    
                    #print(np.exp(-tau_n/2))
                    #nn -= np.exp(-tau_n/2)   # FIXME REMOVE
                    qs_nn = k * ( qL_max - tau_n ) - 2*np.log(nn+1) -  (M-2) * np.log(qL_max / tau_n)   # use qL_max, based on power0 for Df=0
                    #print(k,qs_nn, ( qL_sorted[0] - tau_n ) - 2*np.log(nn+1) -  (M-2) * np.log(qL_sorted.max()/ ((qL_sorted[nn]+qL_sorted[nn+1])/2)) )
                    #qs_nn = k * ( qL_sorted[0] - tau_m ) - 2*np.log(mm+1) -  (M-2) * np.log(qL_sorted.max()/ ((qL_sorted[mm]+qL_sorted[mm+1])/2))
                    #qs_nn = qL_sorted[0] - (qL_sorted[nn]+qL_sorted[nn])/2 - 2*np.log(nn+1) - (M-2) * np.log(qL_sorted.max()/ ((qL_sorted[nn]+qL_sorted[nn])/2))
                except IndexError:        # i.e. n too high; not enough peaks
                    qs_nn = np.nan
                _QS[ni, fTi, si] = qs_nn
                _K[ni, fTi, si] = k

QL = gather_array_by_col(_QL, comm, root)
QS = gather_array_by_col(_QS, comm, root)        
K  = gather_array_by_col(_K, comm, root)   

if rank == root:
    print(np.mean(K, axis=-1), np.std(K, axis=-1))
    np.savez(os.path.join(outfolder, 'q.npz'), QL=QL, QS=QS, fmaxTs=fmaxTs, narr=narr)
