import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interp

def generate_times(len_t, T, n_batch, squash, fbg, fib, seed):
    n_per_batch = int(len_t / n_batch)
    if n_batch == 1 and squash == 1:
        if fbg:
            t = T * np.linspace(0, 1, len_t)
            
            ####t = np.hstack((t[:int(.5*len_t)], np.random.RandomState(seed).uniform(t[int(.5*len_t)],t[-1],int(.5*len_t))))   # 10% random etc
            
            ##t = T * np.linspace(0,1,int(len_t/2))               # 100 uniform than another 100 squashed uniform
            ##t = np.hstack((t, np.linspace(t[-2],t[-1],int(len_t/2))))
        else:
            t = T * np.random.RandomState(seed).rand(len_t)
    else:
        if fbg:
            tmp = np.linspace(0,T,n_batch+1)
            batch_pos = (tmp[1:] + tmp[:-1])/2   # fixed batch grid
        else:
            batch_pos = T * np.random.RandomState(seed).rand(n_batch)
        if fib:
            t = T * np.linspace(0, 1, n_per_batch) / n_batch / squash      # fix inside batch too
            t = np.repeat(t[:,np.newaxis], n_batch, axis=1).T.flatten()    # this gives fllattened [t,t,t,t,t,...]
        else:
            t = T * np.random.RandomState(seed+int(1e8)).rand(len_t) / n_batch / squash
        for i in range(n_batch):
            t[int(i*n_per_batch):int((i+1)*n_per_batch)] += batch_pos[i]
            # FIXME: above starts each batch from batch pos (rather than centering at batch pos). Not a big deal, not sure what is "correct". I'll leave like this. It's random anyway.
    return t

def get_foldername(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T=None):
    folder = "/global/cscratch1/sd/abayer/lookelsewhere/periodogram_astropy/data/"
    if n_batch > 1:
        folder += "lent%d_nb%d_s%d_fbg%d_fib%d_fmin%.2e_pm%s_norm%s_spp%d_ns%.0e_da%d_dk%.2e/" % (len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, dk)
    else:
        folder += "lent%d_nb%d_s%d_fbg%d_fmin%.2e_pm%s_norm%s_spp%d_ns%.0e_da%d_dk%.2e/" % (len_t, n_batch, squash, fbg, fminT, power_method, normalization, spp, Nseeds, dealias, dk)
    
    if nterms > 1:
        folder = folder[:-1] + '_nterms%d_M%d/' % (nterms, M)
    
    if vdPfig1_T is not None:
        folder =  "/global/cscratch1/sd/abayer/lookelsewhere/periodogram_astropy/data/vdPfig1_T%d_fmin%.2e_pm%s_norma%s_spp%d_ns%.0e_da%d/" % (vdPfig1_T, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M) 
    
    if A != 0:
        folder = folder[:-1] + '_A%.2e/' % A
        if peak_removal_df > 0:
            folder = folder[:-1] + 'df%.2e/' % peak_removal_df
            
    if noise_type != 'normal':
        folder = folder[:-1] + '_nt%s/' % noise_type
    
    return folder
    
def get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T=None):
    folder = get_foldername(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)
    file = os.path.join(folder, 'q.npz')
    return file

def get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None):
    title = ''
    if n_batch == 1 and squash == 1:
        title += 'uniform '
    else:
        title += 'grouped '

    if n_batch == 1:
        if fbg:
            title += 'fixed'
        else:
            title += 'random'
    elif n_batch > 1:
        if fbg:
            fbg_str = ' [fix]'
        else:
            fbg_str = ' [rand]'
        if fib:
            fib_str = ' [fix]'
        else:
            fib_str = ' [rand]'
        title += ' nbatch=%d%s, squash=%d%s' % (n_batch, fbg_str, squash, fib_str)
    
    if vdPfig1_T is not None:
        title = 'vdP fig1 (lent=280, T=%d)'%vdPfig1_T 
    
    title += ' (' + power_method + ')'  
    if dealias:
        title += ' dealiased'
        
    title += ', nterms=%d' % nterms
    
    title += ', %s noise' % noise_type
        
    return title

def _gamma(N):
    from scipy.special import gammaln
    # Note: this is closely approximated by (1 - 0.75 / N) for large N
    return np.sqrt(2 / N) * np.exp(gammaln(N / 2) - gammaln((N - 1) / 2))

def fap_baluev(z, W, len_t, normalization='psd'):
    if normalization == 'psd':
        tau = W*np.exp(-z)*np.sqrt(z)
    elif normalization == 'standard':
        dH = 0
        dK = 2
        NH = len_t - dH
        NK = len_t - dK
        z1 = z * NH / 2     # z in this normalization is chi20 - chi2 / chi20 ... convert to Baluev notation
        gammaH = _gamma(NH)
        tau = gammaH * W*(1 - 2*z1/NH)**((NK-1)/2) * np.sqrt(z1)
    elif normalization == 'standard-psd':   # use psd formula for standard (Asymptotic formula)
        dH = 0
        dK = 2
        NH = len_t - dH
        NK = len_t - dK
        z1 = z * NH / 2     # z in this normalization is chi20 - chi2 / chi20 ... convert to Baluev notation
        tau = W*np.exp(-z1)*np.sqrt(z1) 
    return 1 - np.exp(-tau)

def fap_sidak(z):
    return 1 - np.exp(-np.exp(-z))

def qS_sidak(p):
    return -2*np.log(-np.log(1-p))

def plot_CCDF(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk=0, A=0, peak_removal_df=0, noise_type='normal', ax=None, ymin=1e-3, vdPfig1_T=None):
    if ax is None:  # if no ax input just make a single plot of this
        fig, ax = plt.subplots(2,1,figsize=(14,14))
    
    # load data
    info = info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
    narr = info['narr']
    fmaxTs = info['fmaxTs']
    QL = info['QL']
    QS = info['QS']
    if len(fmaxTs) == 5:
        fmaxTs = np.delete(fmaxTs, 4, 0)     # FIXME REMOVE
        QL = np.delete(QL, 4, 0)
        QS = np.delete(QS, 4, 1)
    
    linestyles = ['-','-.','--',':',(0, (1, 10))]#[1:]                                                                                 ########### REMOVE
    for fTi, fmaxT in enumerate(fmaxTs):#[fT for fT in fmaxTs if fT != 500 and fT != 5000]):
        W = fmaxT
        qL = QL[fTi,:]
        #z = qL / 2      # i will work with qL

        # simulation qL
        values, base = np.histogram(qL, bins=100)
        #centers = (base[1:] + base[:-1]) / 2
        #evaluate the cumulative
        cumulative = np.cumsum(values)
        ccdf = 1 - cumulative / len(qL)
        if fTi == 0:
            ax[0].plot(base[1:], ccdf, zorder=1, c='y', linestyle=linestyles[fTi], label=r'$\hat{q}_L$')
        else:
            ax[0].plot(base[1:], ccdf, zorder=1, c='y', linestyle=linestyles[fTi])

        # ax 1 ratio plot baluev
        x = fap_baluev(base[1:]/2, W, len_t, normalization)
        y = ccdf / x
        ax[1].plot(x, y, c='y', linestyle=linestyles[fTi])
        if normalization != 'psd':   # also plot the asymptotic baluev (assuming N>>z>>0)
            x = fap_baluev(base[1:]/2, W, len_t, normalization='%s-psd'%normalization)
            y = ccdf / x
            ax[1].plot(x, y, c='magenta', linestyle=linestyles[fTi])
            if fTi == 0:
                ax[1].plot([], [], zorder=1, color='magenta', linestyle=linestyles[fTi], label=r'asymptotic baluev')

        # baluev
        z = np.linspace(0.1,25,1000)
        fap_b = fap_baluev(z, W, len_t, normalization)
        if fTi == 0:
            ax[0].plot(2*z, fap_b, zorder=1, color='gray', linestyle=linestyles[fTi], label=r'baluev')
        else:
            ax[0].plot(2*z, fap_b, zorder=1, color='gray', linestyle=linestyles[fTi])
        

        # simulation qS
        for ni, n in enumerate(narr):
            qS = QS[ni, fTi, :]
            if np.isnan(qS).any():     # i.e. some realisatios couldn't do self-calibration because not enough peaks
                continue
            values, base = np.histogram(qS, bins=100)
            #evaluate the cumulative
            cumulative = np.cumsum(values)
            ccdf = 1 - cumulative / len(qS)
            if fTi == 0:
                ax[0].plot(base[1:], ccdf, zorder=1, c='C%d'%ni, linestyle=linestyles[fTi], label=r'$\bar{\hat{q}}_S \, (n=%d)$'%n)
            else:
                ax[0].plot(base[1:], ccdf, zorder=1, c='C%d'%ni, linestyle=linestyles[fTi])

            # ax 1 ratio plot
            x = fap_sidak(base[1:]/2)
            y = ccdf / x
            ax[1].plot(x, y, c='C%d'%ni, linestyle=linestyles[fTi])

        # sidak
        z = np.linspace(-10,25,1000)
        fap_s = fap_sidak(z)     #use z_S = q_S/2
        if fTi == 0:
            ax[0].plot(2*z, fap_s, zorder=1, label=r'sidak',color='k')
        else:
            ax[0].plot(2*z, fap_s, zorder=1, color='k')


        # for linestyle legend. put on axis 1 for ease
        ax[1].plot([], [], color='k', linestyle=linestyles[fTi], label=r'$f_{\rm max} T = %.1f$' % fmaxT)

    title = get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None)

    ax[0].set_title(title)

    ax[0].set_xlabel(r'$\hat{q}$')
    ax[0].set_ylabel(r'$P(\hat{Q} \geq \hat{q})$')

    ax[0].set_yscale('log')
    ax[0].set_xlim(-5, 40)
    ax[0].set_ylim(ymin,1.5)

    ax[0].grid(1)
    ax[0].legend(loc='upper right')

    ax[1].hlines(1, ymin, 1, 'k')
    ax[1].set_xlabel(r'$p_{\rm theory}$')
    ax[1].set_ylabel(r'$p/p_{\rm theory}$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(ymin,1)
    ax[1].set_ylim(1e-1,1e1)
    ax[1].legend(loc='upper right')
    
    return ax
    
def plot_CCDF_paper(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk=0, A=0, peak_removal_df=0, noise_type='normal', ax=None, ymin=1e-3, vdPfig1_T=None):
    """
    fbg is ignored. we will plot 1 and 0. I keep in argument to match above function.
    note there's a lot of hard coding and cheating with this to get all the lines and liestyles wanted for the paper...
    """
    if ax is None:  # if no ax input just make a single plot of this
        fig, ax = plt.subplots(2,1,figsize=(11,9), gridspec_kw={'height_ratios': [3, 1]})
    
    # load data
    info_fbg1 = info = dict(np.load(get_filename(len_t, n_batch, squash, 1, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
    info_fbg0 = info = dict(np.load(get_filename(len_t, n_batch, squash, 0, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
    narr = info_fbg1['narr']
    fmaxTs = info_fbg1['fmaxTs'][::-1]               # reverse order (so label ordering is nice)
    assert((narr == info_fbg0['narr']).all())
    assert((fmaxTs == info_fbg0['fmaxTs'][::-1]).all())
    
    QL_1 = info_fbg1['QL'][::-1,:]
    QS_1 = info_fbg1['QS'][:,::-1,:]
    QL_0 = info_fbg0['QL'][::-1,:]
    QS_0 = info_fbg0['QS'][:,::-1,:]
    
    
    #if len(fmaxTs) == 5:
    #    fmaxTs = np.delete(fmaxTs, 4, 0)     # FIXME REMOVE
    #    QL = np.delete(QL, 4, 0)
    #    QS = np.delete(QS, 4, 1)
    
    linestyles = ['--',':',(0, (1, 10))]#[1:]                                                                                 ########### REMOVE
    for fTi, fmaxT in enumerate([fT for fT in fmaxTs if fT != 5000]):
        
        f_nyq = 50    # FIXME: hard coded f_nyq... should eb fine for all my runs (really it's 49.5, but meh).
        W = fmaxT
        qL_1 = QL_1[fTi,:]
        qL_0 = QL_0[fTi,:]

        # simulation qS
        for ni, n in enumerate([n for n in narr if n == 10]):
            qS_1 = QS_1[ni, fTi, :]
            qS_0 = QS_0[ni, fTi, :]
            if np.isnan(qS_1).any() or np.isnan(qS_0).any():     # i.e. some realisatios couldn't do self-calibration because not enough peaks
                continue
            
            if fmaxT <= f_nyq:   # don't plot above nyquist for fixed data
                values_1, base_1 = np.histogram(qS_1, bins=100)
                #evaluate the cumulative
                cumulative_1 = np.cumsum(values_1)
                ccdf_1 = 1 - cumulative_1 / len(qS_1)
                ax[0].plot(base_1[1:], ccdf_1, zorder=1, c='C%d'%ni, linestyle='-', label=r'fixed $[f_{\rm max} = %d]$'%fmaxT)# \, (n=%d)$'%(fmaxT,n))
                
                    
                # ax 1 relative error on qS
                qS_act = base_1[1+9:]
                qS_the = qS_sidak(ccdf_1[9:])
                ax[1].plot(qS_act, np.sqrt(qS_act-np.log(2*np.pi*qS_act))/np.sqrt(qS_the-np.log(2*np.pi*qS_the)), c='C%d'%ni, linestyle='-')

            values_0, base_0 = np.histogram(qS_0, bins=100)
            #evaluate the cumulative
            cumulative_0 = np.cumsum(values_0)
            ccdf_0 = 1 - cumulative_0 / len(qS_0)
            ax[0].plot(base_0[1:], ccdf_0, zorder=1, c='C%d'%ni, linestyle=linestyles[fTi], label=r'random $[f_{\rm max} = %d]$'%fmaxT)# \, (n=%d)$'%(fmaxT,n))

            qS_act = base_0[1+9:]
            qS_the = qS_sidak(ccdf_0[9:])
            ax[1].plot(qS_act, np.sqrt(qS_act-np.log(2*np.pi*qS_act))/np.sqrt(qS_the-np.log(2*np.pi*qS_the)), c='C%d'%ni, linestyle=linestyles[fTi])

    if 1:
        # sidak
        z = np.linspace(-10,25,1000)
        fap_s = fap_sidak(z)     #use z_S = q_S/2
        ax[0].plot(2*z, fap_s, zorder=1, label=r'$1 - \exp{(-e^{-\bar{\hat{q}}_S/2})}$',color='k')


    #title = get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None)

    #ax[0].set_title(title)

    ax[1].set_xlabel(r'$\bar{\hat{q}}_S$')
    ax[0].set_ylabel(r'$P(\bar{\hat{Q}}_S \geq \bar{\hat{q}}_S)$')

    ax[0].set_yscale('log')
    #ax[0].set_xlim(-5, 40)
    ax[0].set_xlim(-5, 20)
    ax[0].set_ylim(ymin,1)#.5)

    ax[0].grid()
    ax[0].legend(loc='upper right')

    ax[1].hlines(1, -5, 20, 'k')
    #ax[1].set_xlabel(r'$\hat{q}_S$')
    ax[1].set_ylabel(r'$\bar{S}/S_{\rm true}$')
    #ax[1].set_xscale('log')
    #ax[1].set_yscale('log')
    ax[1].set_xlim(-5,20)
    ax[1].set_ylim(1-0.2,1+0.05)
    #ax[1].set_yticklabels([],[])
    
    ax[1].grid(1)
    
    plt.savefig('periodogram.pdf', bbox_inches='tight')
    
    return ax
    
def plot_CCDF_LEE3(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dks=[], As=[0], peak_removal_df=0, noise_type='normal', ax=None, ymin=1e-3, vdPfig1_T=None):
    
    if ax is None:  # if no ax input just make a single plot of this
        
        # compute length of narr
        info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dks[0], As[0], peak_removal_df, noise_type, vdPfig1_T)))
        narr = info['narr']
        
        fig, ax = plt.subplots(2,len(narr)+1,figsize=(7*(len(narr)+1),14))
    
    linestyles = [':','-']#'-.','--',':',(0, (1, 10))]#[1:]                                                                                 ########### REMOVE
            
    for Ai,A in enumerate(As):
        for dki,dk in enumerate(dks):
            
            info = info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
            narr = info['narr']
            fmaxTs = info['fmaxTs']
            QL = info['QL']
            QS = info['QS']
            
            # just want nyquist, assume element 0
            for i in range(len(fmaxTs)-1):
                fmaxTs = np.delete(fmaxTs, -1, 0)
                QL = np.delete(QL, -1, 0)
                QS = np.delete(QS, -1, 1)
            
            print('A=',A)
            print('Mean qL = ', np.mean(QL, axis=1))
                
            for fTi, fmaxT in enumerate(fmaxTs):#[fT for fT in fmaxTs if fT != 500 and fT != 5000]):
                W = fmaxT
                qL = QL[fTi,:]
                #z = qL / 2      # i will work with qL

                # simulation qL
                values, base = np.histogram(qL, bins=100)
                #centers = (base[1:] + base[:-1]) / 2
                #evaluate the cumulative
                cumulative = np.cumsum(values)
                ccdf = 1 - cumulative / len(qL)
                if Ai == 1:
                    ax[0,0].plot(base[1:], ccdf, zorder=1, c='C%d'%dki, linestyle=linestyles[Ai], label=r'$\hat{q}_L (\delta k=%.1f)$'%dk)
                else:
                    ax[0,0].plot(base[1:], ccdf, zorder=1, c='C%d'%dki, linestyle=linestyles[Ai])

                # ax 1 ratio plot baluev
                x = fap_baluev(base[1:]/2, W, len_t, normalization)
                y = ccdf / x
                ax[1,0].plot(x, y, c='C%d'%dki, linestyle=linestyles[Ai])
                if normalization != 'psd':   # also plot the asymptotic baluev (assuming N>>z>>0)
                    x = fap_baluev(base[1:]/2, W, len_t, normalization='%s-psd'%normalization)
                    y = ccdf / x
                    ax[1,0].plot(x, y, c='magenta', linestyle=linestyles[Ai])
                    if fTi == 0:
                        ax[1,0].plot([], [], zorder=1, color='magenta', linestyle=linestyles[Ai], label=r'asymptotic baluev')

                # baluev
                z = np.linspace(0.1,25,1000)
                fap_b = fap_baluev(z, W, len_t, normalization)
                if Ai == 0 and dki == 0 and fTi == 0:
                    ax[0,0].plot(2*z, fap_b, zorder=1, color='gray', linestyle='-', label=r'baluev')
                else:
                    ax[0,0].plot(2*z, fap_b, zorder=1, color='gray', linestyle='-')

                
                # simulation qS
                for ni, n in enumerate(narr):
                    qS = QS[ni, fTi, :]
                    if np.isnan(qS).any():     # i.e. some realisatios couldn't do self-calibration because not enough peaks
                        continue
                    values, base = np.histogram(qS, bins=100)
                    #evaluate the cumulative
                    cumulative = np.cumsum(values)
                    ccdf = 1 - cumulative / len(qS)
                    if Ai == 1:
                        ax[0,ni+1].plot(base[1:], ccdf, zorder=1, c='C%d'%dki, linestyle=linestyles[Ai], label=r'$\hat{q}_S (\delta k=%.1f)$'%dk)
                    else:
                        ax[0,ni+1].plot(base[1:], ccdf, zorder=1, c='C%d'%dki, linestyle=linestyles[Ai])

                    # ax 1 ratio plot
                    x = fap_sidak(base[1:]/2)
                    y = ccdf / x
                    ax[1,ni+1].plot(x, y, c='C%d'%dki, linestyle=linestyles[Ai])

                    # sidak
                    z = np.linspace(-10,25,1000)
                    fap_s = fap_sidak(z)     #use z_S = q_S/2
                    if Ai == 0 and dki == 0 and fTi == 0:
                        ax[0,ni+1].plot(2*z, fap_s, zorder=1, label=r'sidak',color='k')
                    else:
                        ax[0,ni+1].plot(2*z, fap_s, zorder=1, color='k')


                    # for linestyle legend. put on axis 1 for ease
                    ###ax[1,ni+1].plot([], [], color='k', linestyle=linestyles[fTi], label=r'$f_{\rm max} T = %.1f$' % fmaxT)

                    ax[0,ni+1].set_title('n=%d'%n, fontsize=20)

    title = get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None) + ' (LEE3)'

    ax[0,0].set_title(title, fontsize=15)

    ax[0,0].set_ylabel(r'$P(\hat{Q} \geq \hat{q})$')
    ax[1,0].set_ylabel(r'$p/p_{\rm theory}$')

    for i in range(len(narr)+1):
        ax[0,i].set_xlabel(r'$\hat{q}$')
        ax[0,i].set_yscale('log')
        ax[0,i].set_ylim(ymin,1.5)

        ax[0,i].grid(1)
        ax[0,i].legend(loc='upper right', fontsize=16)

        ax[1,i].hlines(1, ymin, 1, 'k')
        ax[1,i].set_xlabel(r'$p_{\rm theory}$')
        
        ax[1,i].set_xscale('log')
        ax[1,i].set_yscale('log')
        ax[1,i].set_xlim(ymin,1)
        ax[1,i].set_ylim(1e-1,1e1)
        ax[1,i].legend(loc='upper right')
        
    ax[0,0].set_xlim(-5, 40)
    for j in range(1,len(narr)+1):
        ax[0,j].set_xlim(-5, 40)

                
    plt.savefig('periodogram.pdf', bbox_inches='tight')
    
    return ax

    
def ROC_LEE3(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dks=[], As=[0], peak_removal_df=0, noise_type='normal', ax=None, ymin=1e-3, vdPfig1_T=None):
    
    if ax is None:  # if no ax input just make a single plot of this
        
        # compute length of narr
        info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dks[0], As[0], peak_removal_df, noise_type, vdPfig1_T)))
        narr = info['narr']
        
        fig, ax = plt.subplots(1,len(narr)+1,figsize=(7*(len(narr)+1),7))
    
    linestyles = [':','-']#'-.','--',':',(0, (1, 10))]#[1:]                                                                                 ########### REMOVE
    
    max_qL = -np.inf
    min_qL = np.inf
    max_qS = -np.inf
    min_qS = np.inf
    
    # let's get ROC bins first:
    for Ai,A in enumerate(As):
        for dki,dk in enumerate(dks):
            
            info = info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
            narr = info['narr']
            fmaxTs = info['fmaxTs']
            QL = info['QL']
            QS = info['QS']
            
            # just want nyquist, assume element 0
            for i in range(len(fmaxTs)-1):
                fmaxTs = np.delete(fmaxTs, -1, 0)
                QL = np.delete(QL, -1, 0)
                QS = np.delete(QS, -1, 1)
            
            # FIXME might need to think more carefully about what the nans mean, but for now ignore them all, oterhwise you get bugs in case of batched white noise
            max_qL = max(max_qL, QL[~np.isnan(QL)].max())
            min_qL = min(min_qL, QL[~np.isnan(QL)].min())
            max_qS = max(max_qS, QS[~np.isnan(QS)].max())
            min_qS = min(min_qS, QS[~np.isnan(QS)].min())
                
    Nbins = 1000
    bins_qL = np.linspace(min_qL-1, max_qL+1, Nbins+1)
    bins_qS = np.linspace(min_qS-1, max_qS+1, Nbins+1)
    
    print('bins',bins_qS)
    
    ccdfs = np.empty(shape=(Nbins, len(narr)+1, len(As), len(dks)))   # qL, then qSn
                
    for Ai,A in enumerate(As):
        for dki,dk in enumerate(dks):
            
            info = info = dict(np.load(get_filename(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
            narr = info['narr']
            fmaxTs = info['fmaxTs']
            QL = info['QL']
            QS = info['QS']
            
            # just want nyquist, assume element 0
            for i in range(len(fmaxTs)-1):
                fmaxTs = np.delete(fmaxTs, -1, 0)
                QL = np.delete(QL, -1, 0)
                QS = np.delete(QS, -1, 1)
            
            print('A=',A)
            print('Mean qL = ', np.mean(QL, axis=1))
                
            for fTi, fmaxT in enumerate(fmaxTs):#[fT for fT in fmaxTs if fT != 500 and fT != 5000]):
                W = fmaxT
                qL = QL[fTi,:]
                #z = qL / 2      # i will work with qL

                # simulation qL
                values, base = np.histogram(qL, bins=bins_qL)
                #centers = (base[1:] + base[:-1]) / 2
                #evaluate the cumulative
                cumulative = np.cumsum(values)
                ccdf = 1 - cumulative / len(qL)

                ccdfs[:,0,Ai,dki] = ccdf
                
                # simulation qS
                for ni, n in enumerate(narr):
                    qS = QS[ni, fTi, :]
                    if np.isnan(qS).any():     # i.e. some realisatios couldn't do self-calibration because not enough peaks
                        continue
                    values, base = np.histogram(qS, bins=bins_qS)
                    #evaluate the cumulative
                    cumulative = np.cumsum(values)
                    ccdf = 1 - cumulative / len(qS)
                    
                    ccdfs[:,ni+1,Ai,dki] = ccdf
                    
    A0_id = As.index(0)
    
    for Ai,A in enumerate(As):
        if Ai == A0_id:
            continue
        for dki,dk in enumerate(dks):
            
            ax[0].plot(ccdfs[:, 0, A0_id, dki], ccdfs[:, 0, Ai, dki], c='C%d'%dki, linestyle=linestyles[Ai], label=r'$\hat{q}_L (\delta k=%.1f)$'%dk)
            for ni, n in enumerate(narr):
                if ni > 0:
                    continue   # hacky for aliasing case. FIXME remove
                if dki == 0:   # plot qL line for reference...
                    ax[ni+1].plot(ccdfs[:, 0, A0_id, dki], ccdfs[:, 0, Ai, dki], c='k', linestyle=linestyles[Ai], label=r'$\hat{q}_L$')
                ax[ni+1].plot(ccdfs[:, ni+1, A0_id, dki], ccdfs[:, ni+1, Ai, dki], c='C%d'%dki, linestyle=linestyles[Ai], label=r'$\hat{q}_S (\delta k=%.1f)$'%dk)
                ax[ni+1].set_title('n=%d'%n, fontsize=20)
            
    title = get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None) + ' (LEE3)'

    ax[0].set_title(title, fontsize=15)

    ax[0].set_ylabel(r'TPR')

    for i in range(len(narr)+1):
        ax[i].set_xlabel(r'FPR')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
        #ax[0,i].set_ylim(ymin,1.5)

        ax[i].grid(1)
        ax[i].legend(loc='lower right', fontsize=14)

        #ax[1,i].hlines(1, ymin, 1, 'k')
        #ax[1,i].set_xlabel(r'$p_{\rm theory}$')
        
    
    #ax[0,0].set_xlim(-5, 40)
    #for j in range(1,len(narr)+1):
    #    ax[0,j].set_xlim(-5, 40)

                
    #plt.savefig('periodogram_ROC.pdf', bbox_inches='tight')
    
    return 0#ax
    
    
def p_plot_CCDF_paper(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, ax=None, ymin=1e-3, vdPfig1_T=None):
    """
    fbg is ignored. we will plot 1 and 0. I keep in argument to match above function.
    note there's a lot of hard coding and cheating with this to get all the lines and liestyles wanted for the paper...
    """
    if ax is None:  # if no ax input just make a single plot of this
        fig, ax = plt.subplots(1,2,figsize=(11,9), gridspec_kw={'width_ratios': [3, 1]})
    
    # load data
    info_fbg1 = info = dict(np.load(get_filename(len_t, n_batch, squash, 1, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
    info_fbg0 = info = dict(np.load(get_filename(len_t, n_batch, squash, 0, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, dk, A, peak_removal_df, noise_type, vdPfig1_T)))
    narr = info_fbg1['narr']
    fmaxTs = info_fbg1['fmaxTs'][::-1]               # reverse order (so label ordering is nice)
    assert((narr == info_fbg0['narr']).all())
    assert((fmaxTs == info_fbg0['fmaxTs'][::-1]).all())
    
    QL_1 = info_fbg1['QL'][::-1,:]
    QS_1 = info_fbg1['QS'][:,::-1,:]
    QL_0 = info_fbg0['QL'][::-1,:]
    QS_0 = info_fbg0['QS'][:,::-1,:]
    
    
    #if len(fmaxTs) == 5:
    #    fmaxTs = np.delete(fmaxTs, 4, 0)     # FIXME REMOVE
    #    QL = np.delete(QL, 4, 0)
    #    QS = np.delete(QS, 4, 1)
    
    linestyles = ['--',':',(0, (1, 10))]#[1:]                                                                                 ########### REMOVE
    for fTi, fmaxT in enumerate([fT for fT in fmaxTs if fT != 5000]):
        
        f_nyq = 50    # FIXME: hard coded f_nyq... should eb fine for all my runs (really it's 49.5, but meh).
        W = fmaxT
        qL_1 = QL_1[fTi,:]
        qL_0 = QL_0[fTi,:]
        
        """
        # simulation qL
        # first fbg 1
        if fmaxT <= f_nyq:   # don't plot above nyquist for fixed data
            values, base_1 = np.histogram(qL_1, bins=100)
            #centers = (base[1:] + base[:-1]) / 2
            #evaluate the cumulative
            cumulative = np.cumsum(values)
            ccdf_1 = 1 - cumulative / len(qL_1)
            if fTi == 0:
                ax[0].plot(base_1[1:], ccdf_1, zorder=1, c='y', linestyle=linestyles[fTi], label=r'$\hat{q}_L$ (fixed)')
            else:
                ax[0].plot(base_1[1:], ccdf_1, zorder=1, c='y', linestyle=linestyles[fTi])
            
            # ax 1 ratio plot
            x = fap_baluev(base_1[1:]/2, W, len_t, normalization)
            y = ccdf_1 / x
            ax[1].plot(x, y, c='y', linestyle=linestyles[fTi])

         # first fbg 0
        values, base_0 = np.histogram(qL_0, bins=100)
        #centers = (base[1:] + base[:-1]) / 2
        #evaluate the cumulative
        cumulative = np.cumsum(values)
        ccdf_0 = 1 - cumulative / len(qL_0)
        if fTi == 0:
            ax[0].plot(base_0[1:], ccdf_0, zorder=1, c='y', linestyle=linestyles[fTi], label=r'$\hat{q}_L$ (random)')
        else:
            ax[0].plot(base_0[1:], ccdf_0, zorder=1, c='y', linestyle=linestyles[fTi])
        
        # ax 1 ratio plot
        x = fap_baluev(base_0[1:]/2, W, len_t, normalization)
        y = ccdf_0 / x
        ax[1].plot(x, y, c='y', linestyle=linestyles[fTi])
        
        # baluev
        z = np.linspace(0.1,25,1000)
        fap_b = fap_baluev(z, W, len_t, normalization)
        if fTi == 0:
            ax[0].plot(2*z, fap_b, zorder=1, color='gray', linestyle=linestyles[fTi], label=r'baluev')
        else:
            ax[0].plot(2*z, fap_b, zorder=1, color='gray', linestyle=linestyles[fTi])
        """

        # simulation qS
        for ni, n in enumerate([n for n in narr if n == 10]):
            qS_1 = QS_1[ni, fTi, :]
            qS_0 = QS_0[ni, fTi, :]
            if np.isnan(qS_1).any() or np.isnan(qS_0).any():     # i.e. some realisatios couldn't do self-calibration because not enough peaks
                continue
            
            if fmaxT <= f_nyq:   # don't plot above nyquist for fixed data
                values_1, base_1 = np.histogram(qS_1, bins=100)
                #evaluate the cumulative
                cumulative_1 = np.cumsum(values_1)
                ccdf_1 = 1 - cumulative_1 / len(qS_1)
                ax[0].plot(base_1[1:], ccdf_1, zorder=1, c='C%d'%ni, linestyle='-', label=r'fixed $[f_{\rm max} = %d]$'%fmaxT)# \, (n=%d)$'%(fmaxT,n))
                
                    
                # ax 1 ratio plot
                x = fap_sidak(base_1[1:]/2)
                y = x / ccdf_1
                ax[1].plot(y, ccdf_1, c='C%d'%ni, linestyle='-')

            values_0, base_0 = np.histogram(qS_0, bins=100)
            #evaluate the cumulative
            cumulative_0 = np.cumsum(values_0)
            ccdf_0 = 1 - cumulative_0 / len(qS_0)
            ax[0].plot(base_0[1:], ccdf_0, zorder=1, c='C%d'%ni, linestyle=linestyles[fTi], label=r'random $[f_{\rm max} = %d]$'%fmaxT)# \, (n=%d)$'%(fmaxT,n))

            x = fap_sidak(base_0[1:]/2)
            y = x / ccdf_0
            ax[1].plot(y, ccdf_0, c='C%d'%ni, linestyle=linestyles[fTi])

    if 1:
        # sidak
        z = np.linspace(-10,25,1000)
        fap_s = fap_sidak(z)     #use z_S = q_S/2
        ax[0].plot(2*z, fap_s, zorder=1, label=r'$1 - \exp{(-e^{-\bar{\hat{q}}_S/2})}$',color='k')


    #title = get_title(len_t, n_batch, squash, fbg, fib, fminT, power_method, normalization, spp, Nseeds, dealias, nterms, M, noise_type, vdPfig1_T=None)

    #ax[0].set_title(title)

    ax[0].set_xlabel(r'$\bar{\hat{q}}_S$')
    ax[0].set_ylabel(r'$P(\bar{\hat{Q}}_S \geq \bar{\hat{q}}_S)$')

    ax[0].set_yscale('log')
    #ax[0].set_xlim(-5, 40)
    ax[0].set_xlim(-5, 20)
    ax[0].set_ylim(ymin,1)#.5)

    ax[0].grid()
    ax[0].legend(loc='upper right')

    ax[1].vlines(1, ymin, 1, 'k')
    ax[1].set_ylabel(r'$P_{\rm true}$')
    ax[1].set_xlabel(r'$\bar{P}/P_{\rm true}$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(ymin,1)
    ax[1].set_xlim(1e-1,1e1)
    ax[1].set_yticklabels([],[])
    
    ax[1].grid()
    
    plt.savefig('periodogram.pdf', bbox_inches='tight')

def read_peak_info_astropy(dirname, seed, fmaxT, nda=None):
    if nda is not None:
        fname = dirname + '/seed' + str(seed).zfill(6) + '/peak_locs_info_fmaxT%.2e_nda%d.npz' % (fmaxT, nda)
    else:
        fname = dirname + '/seed' + str(seed).zfill(6) + '/peak_locs_info_fmaxT%.2e.npz' % fmaxT
    pl = dict(np.load(fname))
    return pl['freq'], pl['power'], pl['theta'], pl['peak_indices'], pl['f_fit'], pl['theta_fit']
