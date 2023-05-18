import matplotlib.pyplot as plt
import numpy as np
import os


def plt_Vretrieval_synthetic(PFT_uni,output_directory, Expname,
                             Anobs_trn  ,Anobs_tst ,
                             Anpred_trn ,Anpred_tst,
                             vcmax_CLM45, vcmax_sim):
    barWidth = 0.25

    br1 = np.arange(len(PFT_uni))
    br2 = [x + barWidth for x in br1]

    fig, axs = plt.subplots(1,2, figsize=(12, 5.5))
    fig.suptitle('Synthetic_Case (Single Parameter Recovery)', fontsize = 15, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.95, wspace=0.15, bottom = 0.23)
    axs[0].bar(br1, vcmax_CLM45, color ='g', width = barWidth,
            edgecolor ='black', label ='CLM4.5')
    axs[0].bar(br2, vcmax_sim, color ='b', width = barWidth,
            edgecolor ='black', label ='Recovered')
    axs[0].set_xticks([r+barWidth/2  for r in range(len(PFT_uni))])
    axs[0].set_xticklabels(list(PFT_uni), rotation = 70, fontsize = 11)
    axs[0].set_ylabel(r'V$_{\rm c,max25}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[0].tick_params(labelsize = 11,  axis='both', which='major', pad=0.0)
    axs[0].set_title(r'(a) Maximum Carboxylation Rate at 25$^\circ$C', fontsize = 13)
    axs[0].legend()

    axs[1].plot(np.arange(0, 85), np.arange(0, 85), linestyle='dashed', color = "black",zorder = 0 )
    axs[1].scatter( Anobs_trn,Anpred_trn, color = "green", label = "Train", edgecolor = 'black', s= 90, linewidth = 0.5 )
    axs[1].scatter( Anobs_tst,Anpred_tst, color = "blue", label = "Test", edgecolor = 'black', s = 90, linewidth = 0.5)
    axs[1].set_title(r'(b) Net Photosynthesis Rate', fontsize = 13)
    axs[1].set_xlabel(r'Actual A$_{\rm n}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[1].set_ylabel(r'Simulated A$_{\rm n}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[1].tick_params(labelsize = 11)
    axs[1].legend()
    plt.savefig(os.path.join(output_directory, "{}.PNG".format(Expname)))

    return

def plt_VBretrieval_synthetic(PFT_uni    ,output_directory, Expname,
                              btran_trn  ,btran_tst ,btran_trn_syn , btran_tst_syn,
                              B_trn      ,B_tst     ,B_trn_syn     , B_tst_syn    ,
                              Anobs_trn  , Anobs_tst,Anpred_trn    , Anpred_tst   ,
                              vcmax_CLM45, vcmax_sim):
    # Vcmax25
    barWidth = 0.25
    br1 = np.arange(len(PFT_uni))
    br2 = [x + barWidth for x in br1]

    # Btran
    # recovered btran train and test
    btran_trn_rec = btran_trn.to('cpu')
    btran_tst_rec = btran_tst.to('cpu')

    # B
    # recovered B train and test
    B_trn_rec = B_trn.to('cpu')
    B_tst_rec = B_tst.to('cpu')

    fig, axs = plt.subplots(2,2, figsize=(12, 10.5))
    fig.suptitle('Synthetic_Case (Dual Parameter Recovery)', fontsize = 15, fontweight='bold')
    plt.subplots_adjust(left=0.07, right=0.95, wspace=0.15, hspace = 0.42, top = 0.92, bottom = 0.05)
    axs[0,0].bar(br1, vcmax_CLM45, color ='g', width = barWidth,
            edgecolor ='black', label ='CLM4.5')
    axs[0,0].bar(br2, vcmax_sim, color ='b', width = barWidth,
            edgecolor ='black', label ='Recovered')
    axs[0,0].set_xticks([r + barWidth/2 for r in range(len(PFT_uni))])
    axs[0,0].set_xticklabels(list(PFT_uni), rotation = 70, fontsize = 11)
    axs[0,0].set_ylabel(r'V$_{\rm c,max25}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[0,0].tick_params(labelsize = 11,  axis='both', which='major', pad=0.0)
    axs[0,0].set_title(r'(a) Maximum Carboxylation Rate at 25$^\circ$C', fontsize = 13 )
    axs[0,0].legend()


    axs[1,0].scatter( btran_trn_syn, btran_trn_rec,color = "green", label = "Train", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[1,0].scatter( btran_tst_syn, btran_tst_rec, color = "blue", label = "Test", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[1,0].plot(np.linspace(0.4, 1), np.linspace(0.4, 1), linestyle='dashed', color = "black", zorder = 0)
    axs[1,0].set_title(r'(c) Soil Water Stress ($\beta_{\rm t}$)', fontsize = 13 )
    axs[1,0].set_xlabel('Actual' ,fontsize = 11)
    axs[1,0].set_ylabel('Recovered' ,fontsize = 11)
    axs[1,0].tick_params(labelsize = 11)
    axs[1,0].legend()


    axs[0,1].scatter( B_trn_syn, B_trn_rec, color = "green", label = "Train", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[0,1].scatter( B_tst_syn, B_tst_rec, color = "blue", label = "Test", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[0,1].plot(np.linspace(0.1, 0.45), np.linspace(0.1, 0.45), linestyle='dashed', color = "black", zorder = 0)
    axs[0,1].set_xlabel('Actual', fontsize =11)
    axs[0,1].set_ylabel('Recovered', fontsize = 11)
    axs[0,1].set_title('(b) Parameter B', fontsize = 13)
    axs[0,1].tick_params(labelsize = 11)
    axs[0,1].legend()

    axs[1,1].scatter( Anobs_trn,Anpred_trn, color = "green", label = "Train", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[1,1].scatter( Anobs_tst,Anpred_tst, color = "blue", label = "Test", s = 90, edgecolor = 'black', linewidth = 0.5)
    axs[1,1].plot(np.arange(0, 85), np.arange(0, 85), linestyle='dashed', color = "black", zorder = 0)
    axs[1,1].set_xlabel(r'Actual A$_{\rm n}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[1,1].set_ylabel(r'Simulated A$_{\rm n}$ ($\mu$mol m$^{-2}$ s$^{-1})$', fontsize = 11)
    axs[1,1].set_title(r'(d) Net Photosynthesis Rate', fontsize = 13)
    axs[1,1].tick_params(labelsize = 11)
    axs[1,1].legend()
    plt.savefig(os.path.join(output_directory, "{}.PNG".format(Expname)))
    return

