"""Generate every figure of the manuscript. Run:  python make_figures.py   (after verify_numerics.py, analyses_*.py, sens.py)"""
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from malaria_model import *
from timing import WINDOWS
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
figure_directory = Path(__file__).resolve().parents[1] / "figs"
figure_directory.mkdir(exist_ok=True)
q=derived(P)
def save(name):
    plt.savefig(figure_directory / f"{name}.pdf")
    plt.savefig(figure_directory / f"{name}.png", dpi=200)
    plt.close()
OBS='#444444'; C=['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b']
# ---------------------------------------------------------------- surveillance values from Hailemariam et al. (2025)
yr=np.arange(2014,2025)
pop=[5072461,5180386,5288310,5396235,5504160,5612084,5720009,5827934,5935859,6043783,6194878]
cases=np.array([251294,213902,301405,187720,154072,187266,123793,19110,61196,302605,382955])
ir=[49.54,41.29,56.99,34.79,27.99,33.37,21.64,3.28,10.31,50.07,61.82]
inp=[1275,1395,1891,1296,812,715,355,37,147,756,1874]; deaths=np.array([89,65,77,52,48,31,24,11,98,26,76]); cfr=[0.04,0.03,0.03,0.03,0.03,0.02,0.02,0.06,0.16,0.01,0.02]
def shade(ax): ax.axvspan(2020.83,2022.83,color='0.85',zorder=0,label='Conflict period (Nov 2020–Nov 2022)')
# Fig 1
fig,ax=plt.subplots(figsize=(6.4,3.4)); shade(ax); ax.bar(yr,cases/1e3,color=OBS,width=0.7,label='Reported cases (surveillance)')
ax.set_xlabel('Year'); ax.set_ylabel('Reported malaria cases (thousands)'); ax.set_xticks(yr); ax.set_title('Observed surveillance data (not model output)'); ax.legend(frameon=False,loc='upper left'); save('figure1_annual_cases')
# Fig 2
fig,ax=plt.subplots(figsize=(4.6,3.4)); sp=[1426907,758411]; b=ax.bar([r'$\mathit{P.\ falciparum}$',r'$\mathit{P.\ vivax}$'],[s/1e6 for s in sp],color=[C[1],C[0]],width=0.55)
for r,s in zip(b,sp): ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.02,f"{s:,}\n({100*s/sum(sp):.1f}%)",ha='center',fontsize=8)
ax.set_ylabel('Confirmed cases 2014–2024 (millions)'); ax.set_ylim(0,1.8); ax.set_title('Observed species composition (n = 2,185,318)'); save('figure2_species_composition')
# Fig 4: annual surveillance burden
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4))
for ax in axs.ravel(): shade(ax); ax.set_xticks(yr[::2])
axs[0,0].bar(yr,cases/1e3,color=OBS); axs[0,0].set_ylabel('Cases (thousands)'); axs[0,0].set_title('(a) Reported cases')
axs[0,1].plot(yr,ir,'o-',color=OBS); axs[0,1].set_ylabel('Incidence per 1,000'); axs[0,1].set_title('(b) Reported incidence rate')
axs[1,0].bar(yr,inp,color=OBS); axs[1,0].set_ylabel('Inpatient cases'); axs[1,0].set_title('(c) Reported inpatient cases')
axs[1,1].bar(yr,deaths,color=OBS); axs[1,1].set_ylabel('Deaths'); axs[1,1].set_title('(d) Reported deaths (line: CFR %)'); a2=axs[1,1].twinx(); a2.plot(yr,cfr,'o-',color=C[1]); a2.set_ylabel('CFR (%)'); a2.spines['top'].set_visible(False)
for ax in axs[1]: ax.set_xlabel('Year')
fig.suptitle('Observed surveillance data (Hailemariam et al., 2025)',y=1.0,fontsize=10); plt.tight_layout(); save('figure4_annual_surveillance_burden')
# ---------------------------------------------------------------- simulations
t=np.arange(366.0); z=np.zeros_like(t); Rt=R_cap(t)
def sim(a,b,qq=q,seasonal=True):
    _,X=rk4_forward(qq,a,b,1.0,seasonal); return X
Xref=sim(z+0.05,z); Nh=Xref[:,:4].sum(1)
# Fig 5: baseline dynamics
fig,axs=plt.subplots(2,2,figsize=(7.4,5.4))
axs[0,0].plot(t,1000*Xref[:,2]/Nh,color=C[1]); axs[0,0].set_title('(a) Simulated infectious prevalence'); axs[0,0].set_ylabel('Infectious humans per 1,000')
for j,(nm,c) in enumerate(zip(['$S_h$','$E_h$','$I_h$','$R_h$'],[C[0],C[3],C[1],C[2]])): axs[0,1].semilogy(t[1:],np.maximum(Xref[1:,j]/Nh[1:],1e-9),color=c,label=nm)
axs[0,1].set_ylim(1e-6,2); axs[0,1].set_title('(b) Simulated human compartments'); axs[0,1].set_ylabel('Proportion of $N_h$ (log scale)'); axs[0,1].legend(frameon=False,ncol=2)
Nv=Xref[:,4:].sum(1); axs[1,0].plot(t,Nv/q['Nv0'],color=C[4],label='$N_v/N_{v0}$'); axs[1,0].plot(t,Xref[:,6]/q['Nv0']*20,color=C[3],label='$20\\,I_v/N_{v0}$')
axs[1,0].set_title('(c) Simulated vector abundance'); axs[1,0].legend(frameon=False); axs[1,0].set_ylabel('Normalised abundance')
axs[1,1].plot(t,1+q['phi']*np.sin(2*np.pi*(t-q['sigma'])/365),color=C[4]); axs[1,1].axvline(149,ls=':',color='0.5'); axs[1,1].axvline(331,ls=':',color='0.5')
axs[1,1].set_title('(d) Assumed recruitment multiplier'); axs[1,1].set_ylabel('$\\Lambda_v(t)/\\bar\\Lambda_v$')
for ax in axs[1]: ax.set_xlabel('Day of simulation year (0 = 1 January)')
for ax in axs[0]: ax.set_xlabel('Day of simulation year (0 = 1 January)')
plt.tight_layout(); save('figure5_baseline_dynamics')
# Fig 6: feasible scenarios
sc=json.load(open('analysis_scen.json'))['scenarios']; r=fbs(q,dt=1.0)
curves={'No intervention':(z,z),'Minimal reference':(z+0.05,z),'LLIN only (0.50)':(z+0.5,z),'IRS only (0.417)':(z,z+0.5/1.2),'LLIN+IRS split':(z+0.25,z+0.25/1.2),'Computed control':(r['u1'],r['u2'])}
fig,axs=plt.subplots(2,2,figsize=(7.4,5.6))
for (nm,(a,b)),c in zip(curves.items(),C):
    X=sim(a,b); Nh_=X[:,:4].sum(1); axs[0,0].plot(t,1000*X[:,2]/Nh_,color=c,label=nm); axs[0,1].plot(t,1000*X[:,1]/Nh_,color=c)
axs[0,0].set_title('(a) Simulated infectious prevalence'); axs[0,0].set_ylabel('per 1,000'); axs[0,0].legend(frameon=False,fontsize=7)
axs[0,1].set_title('(b) Simulated exposed prevalence'); axs[0,1].set_ylabel('per 1,000')
for ax in axs[0]: ax.set_xlabel('Day')
lab=['None','Minimal','LLIN\nonly','IRS\nonly','LLIN+IRS\nsplit','Computed\ncontrol']
axs[1,0].bar(lab,[s['peak'] for s in sc],color=C); axs[1,0].set_title('(c) Peak infectious prevalence'); axs[1,0].set_ylabel('per 1,000')
axs[1,1].bar(lab,[s['BI'] for s in sc],color=C); axs[1,1].set_title('(d) Cumulative infectious burden'); axs[1,1].set_ylabel('infectious person-days per 1,000')
for ax in axs[1]: ax.tick_params(axis='x',labelsize=7)
fig.suptitle('Resource-feasible scenarios (resource use $\\leq 0.5 \\leq R(t)$ for all fixed scenarios)',y=1.0,fontsize=9.5); plt.tight_layout(); save('figure6_fixed_intervention_comparison')
# Fig 7: computed control
fig,axs=plt.subplots(2,2,figsize=(7.4,5.6)); ti=t[1:-1]
axs[0,0].plot(ti,r['u1'][1:-1],color=C[0],label='$u_1$ (LLIN)'); axs[0,0].plot(ti,r['u2'][1:-1],color=C[1],label='$u_2$ (IRS)'); axs[0,0].set_ylim(-0.02,0.9); axs[0,0].legend(frameon=False)
axs[0,0].set_title('(a) Computed controls (interior nodes)'); axs[0,0].set_ylabel('Control level')
axs[0,1].plot(t,Rt,'k--',label='Capacity $R(t)$'); axs[0,1].plot(ti,r['use'][1:-1],color=C[2],label='Use $c_1u_1+c_2u_2$ (interior nodes)'); axs[0,1].legend(frameon=False); axs[0,1].set_title('(b) Resource use vs capacity'); axs[0,1].set_ylim(0,0.9)
axs[1,0].semilogy(np.arange(1,len(r['hist'])+1),r['hist'],color=C[4]); axs[1,0].axhline(1e-6,ls='--',color='k'); axs[1,0].set_title('(c) Sweep convergence $\\Delta_k$'); axs[1,0].set_xlabel('Iteration $k$'); axs[1,0].set_ylabel('$\\Delta_k$')
Xo=sim(r['u1'],r['u2']); axs[1,1].plot(t,1000*Xref[:,2]/Nh,color=C[1],label='Minimal reference'); axs[1,1].plot(t,1000*Xo[:,2]/Xo[:,:4].sum(1),color=C[0],label='Computed control'); axs[1,1].legend(frameon=False)
axs[1,1].set_title('(d) Simulated infectious prevalence'); axs[1,1].set_ylabel('per 1,000'); axs[1,1].set_xlabel('Day')
for ax in axs[0]: ax.set_xlabel('Day')
plt.tight_layout(); save('figure7_optimal_control')
# Fig 9: PRCC
S=json.load(open('sens.json')); lbl={'a':'$a$','beta_h':'$\\beta_h$','beta_v':'$\\beta_v$','mv':'$N_{v0}/N_h$','gamma_h':'$\\gamma_h$','omega':'$\\omega$','pd':'$p_d$','eps1':'$\\epsilon_1$','phi':'$\\phi$','tau_h':'$\\tau_h$','mu_v':'$\\mu_v$','alpha_h':'$\\alpha_h$','alpha_v':'$\\alpha_v$'}
fig,axs=plt.subplots(1,2,figsize=(7.6,4.2))
for ax,key,ttl in [(axs[0],'prcc_R0','(a) $\\mathcal{R}_0$ (autonomous model)'),(axs[1],'prcc_peak','(b) Peak infectious prevalence (seasonal model)')]:
    d=S[key]; ks=sorted(d,key=lambda k:abs(d[k])); ax.barh([lbl[k] for k in ks],[d[k] for k in ks],color=[C[1] if d[k]>0 else C[0] for k in ks]); ax.axvline(0,color='k',lw=0.6); ax.set_xlim(-1,1); ax.set_title(ttl); ax.set_xlabel('PRCC')
plt.tight_layout(); save('figure9_sensitivity_prcc')
# Fig 10: timing
T=json.load(open('analysis_scen.json'))['timing']; B=json.load(open('analysis_burnin.json'))['rows']
fig,axs=plt.subplots(1,2,figsize=(7.8,3.8))
wn=['W1\n1–90','W2\n91–180','W3\n181–270','W4\n276–365']; wins=WINDOWS.values()
axs[0].plot(t,1+q['phi']*np.sin(2*np.pi*(t-q['sigma'])/365),color=C[4],label='Recruitment multiplier')
for i_,((s,e),c) in enumerate(zip(wins,C)):
    axs[0].axvspan(s,e,ymin=0.0,ymax=0.12,color=c,alpha=0.8); axs[0].text((s+e)/2,0.575,f'W{i_+1}',ha='center',va='center',fontsize=8,color='white')
axs[0].axvline(331,ls=':',color='0.4'); axs[0].set_xlabel('Day'); axs[0].set_title('(a) High-effort windows (0.30, 1/6); use = 0.50 ≤ R(t)'); axs[0].set_ylabel('Multiplier')
x=np.arange(4); w=0.38
axs[1].bar(x-w/2,[T[i+1]['red_BI'] for i in range(4)],w,color=C[0],label='From initial seed'); axs[1].bar(x+w/2,[B[i]['red_BI'] for i in range(4)],w,color=C[2],label='From quasi-periodic state')
axs[1].axhline(T[-1]['red_BI'],ls='--',color=C[0],lw=1,label='Same effort spread over year (seed)'); axs[1].axhline(B[4]['red_BI'],ls='--',color=C[2],lw=1,label='Same effort spread over year (quasi-periodic)')
axs[1].set_xticks(x); axs[1].set_xticklabels(wn); axs[1].set_ylabel('Reduction in cumulative burden (%)'); axs[1].set_title('(b) Simulated effect of a 90-day window'); axs[1].legend(frameon=False,fontsize=7)
plt.tight_layout(); save('figure10_timing_strategies')
# Fig 3: bifurcation
BD=np.load('bifurcation_curve.npy'); rp=json.load(open('rp_curve.json'))
fig,axs=plt.subplots(1,2,figsize=(7.6,3.6))
axs[0].plot(BD[:,1],1000*BD[:,3],color=C[1]); axs[0].axvline(1,ls='--',color='k',lw=0.8); axs[0].set_xlabel('$\\mathcal{R}_0$ (varied through biting rate $a$)'); axs[0].set_ylabel('Endemic $I_h^*/N_h^*$ per 1,000'); axs[0].set_title('(a) Forward bifurcation at $\\mathcal{R}_0=1$')
baseline = json.load(open('analysis_A_B.json'))
axs[0].plot(baseline['R0_u0'], 1000*baseline['endemic_IhNh'], 'ko') 
axs[1].plot(rp['R0'],rp['Rp'],'o-',color=C[0]); lim=[0.4,2.2]; axs[1].plot(lim,lim,'k--',lw=0.8); axs[1].axhline(1,color='0.6',lw=0.6); axs[1].axvline(1,color='0.6',lw=0.6)
axs[1].set_xlabel('$\\mathcal{R}_0$ (constant recruitment)'); axs[1].set_ylabel('$\\mathcal{R}_p$ (seasonal recruitment, $\\phi=0.45$)'); axs[1].set_title('(b) Autonomous vs periodic threshold')
plt.tight_layout(); save('figure3_bifurcation_threshold')
# Fig 8: numerical verification
V=json.load(open('verify_results.json')); Hn=np.load('verify_hist.npz')
fig,axs=plt.subplots(2,2,figsize=(7.6,5.8))
def _lab(k):
    if k.startswith('theta'): return '$\\theta$='+k.split('_')[1]+', $u^{(0)}$=(0.05,0)'
    a,b=k.split('_'); return '$\\theta$=0.15, $u^{(0)}$=('+a+','+b+')'
for k in Hn.files: axs[0,0].semilogy(np.arange(1,len(Hn[k])+1),Hn[k],lw=1,label=_lab(k))
axs[0,0].axhline(1e-6,ls='--',color='k'); axs[0,0].set_xlabel('Iteration $k$'); axs[0,0].set_ylabel('$\\Delta_k$'); axs[0,0].set_title('(a) Initialisation and relaxation'); axs[0,0].legend(frameon=False,fontsize=5.5,ncol=1)
R_=V['refinement']; dts=[r_['dt'] for r_ in R_[:-1]]; axs[0,1].loglog(dts,[max(r_['relJ'],1e-8) for r_ in R_[:-1]],'o-',color=C[0]); axs[0,1].set_xlabel('Time step $\\Delta t$ (days)'); axs[0,1].set_ylabel('Relative change in $J$ vs $\\Delta t=0.125$'); axs[0,1].set_title('(b) Time-step refinement'); axs[0,1].set_xticks(dts); axs[0,1].set_xticklabels([str(d_) for d_ in dts]); axs[0,1].minorticks_off()
cp=V['comparators']; ks=list(cp); vals=[cp[k]/cp['optimal (FBS)'] for k in ks]; axs[1,0].barh(['Computed candidate' if k=='optimal (FBS)' else k for k in ks],vals,color=[C[0] if k!='optimal (FBS)' else C[2] for k in ks]); axs[1,0].set_xscale('log'); axs[1,0].set_xlim(0.95,7); axs[1,0].set_xticks([1,2,4]); axs[1,0].set_xticklabels(['1','2','4']); axs[1,0].axvline(1,color='k',lw=0.6); axs[1,0].set_xlabel('$J/J_{\\mathrm{FBS}}$ (feasible comparators)'); axs[1,0].set_title('(c) Objective vs feasible comparators'); axs[1,0].tick_params(axis='y',labelsize=6.5)
g=V['gradcheck']; axs[1,1].bar(np.arange(len(g)),[x_['rel'] for x_ in g],color=C[4]); axs[1,1].set_yscale('log'); axs[1,1].set_xticks(np.arange(len(g))); axs[1,1].set_xlabel('Random direction'); axs[1,1].set_ylabel('Relative error (adjoint vs finite difference)'); axs[1,1].set_title('(d) Adjoint gradient check')
plt.tight_layout(); save('figure8_numerical_verification')
print("figures written")
