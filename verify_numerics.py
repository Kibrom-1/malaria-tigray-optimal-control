"""Numerical verification of the constrained forward-backward sweep (Reviewer 4, major comment 2)."""
import numpy as np, json, time
from malaria_model import *
q=derived(P); out={}
def interp_to(res, tgrid):
    return np.interp(tgrid,res['t'],res['u1']), np.interp(tgrid,res['t'],res['u2'])

# ---- (1) time-step refinement (RK4 + KKT sweep), reference = finest grid
dts=[5.0,2.5,1.0,0.5,0.25,0.125]; runs={}
for dt in dts:
    t0=time.time(); r=fbs(q,dt=dt); runs[dt]=r
    print(f"dt={dt}: conv={r['converged']} it={r['iters']} J={r['J']:.6e} ({time.time()-t0:.1f}s)")
ref=runs[0.125]; tg=np.arange(5,361.0,1.0)   # exclude boundary layers: first/last node are boundary artefacts (see text)
rows=[]
for dt in dts:
    r=runs[dt]; u1,u2=interp_to(r,tg); ru1,ru2=interp_to(ref,tg)
    o=outcomes(r['t'],r['X'])
    rows.append(dict(dt=dt,iters=r['iters'],converged=bool(r['converged']),J=r['J'],
        relJ=abs(r['J']-ref['J'])/ref['J'],du1=float(np.abs(u1-ru1).max()),du2=float(np.abs(u2-ru2).max()),
        mean_u1=float(r['u1'].mean()),mean_u2=float(r['u2'].mean()),peak=float(o['peak']),BI=float(o['BI']),
        max_viol=float((r['use']-r['Rt']).max())))
out['refinement']=rows
# observed order from J differences (dt=2,1,0.5 vs 0.25)
d=[abs(runs[dt]['J']-runs[0.125]['J']) for dt in dts[:-1]]
out['J_diffs']=d
# ---- (2) initialisation / relaxation independence at dt=1
inits={"(0.05,0)":(0.05,0.0),"(0,0)":(0.0,0.0),"(0.50,0)":(0.5,0.0),"(0.25,0.20)":(0.25,0.20),"(0,0.40)":(0.0,0.40)}
base=runs[1.0]; irows=[]; hists={}
for name,ui in inits.items():
    r=fbs(q,dt=1.0,u_init=ui); hists[name]=r['hist']
    irows.append(dict(init=name,theta=0.15,iters=r['iters'],conv=bool(r['converged']),J=r['J'],
        relJ=abs(r['J']-base['J'])/base['J'],du1=float(np.abs(r['u1']-base['u1']).max()),du2=float(np.abs(r['u2']-base['u2']).max())))
for th in [0.05,0.30,0.50]:
    r=fbs(q,dt=1.0,theta=th,maxit=600); hists[f"theta={th}"]=r['hist']
    irows.append(dict(init="(0.05,0)",theta=th,iters=r['iters'],conv=bool(r['converged']),J=r['J'],
        relJ=abs(r['J']-base['J'])/base['J'],du1=float(np.abs(r['u1']-base['u1']).max()),du2=float(np.abs(r['u2']-base['u2']).max())))
out['init']=irows
for r_ in irows: print(r_)

# ---- (3) adjoint / gradient consistency check (finite differences of reduced objective)
t=base['t']; u1=base['u1'].copy(); u2=base['u2'].copy()
rng=np.random.default_rng(42)
w=np.ones_like(t)*1.0; w[0]=w[-1]=0.5
def Jof(a,b):
    _,X=rk4_forward(q,a,b,1.0); return objective(t,X,a,b,q)
# unconstrained reduced gradient dJ/du_i(t_k) ~ w_k*(B_i u_i + g_i)   (eta not included)
g1,g2=stationary_terms_vec(base['X'],base['PSI'],q)
grad1=w*(q['B1']*u1+g1); grad2=w*(q['B2']*u2+g2)
gc=[]
for trial in range(5):
    d1=rng.normal(size=t.size); d2=rng.normal(size=t.size)
    # smooth the direction
    k=np.ones(15)/15; d1=np.convolve(d1,k,'same'); d2=np.convolve(d2,k,'same')
    d1/=np.abs(d1).max(); d2/=np.abs(d2).max()
    h=1e-3
    fd=(Jof(u1+h*d1,u2+h*d2)-Jof(u1-h*d1,u2-h*d2))/(2*h)
    an=np.sum(grad1*d1)+np.sum(grad2*d2)
    gc.append((fd,an,abs(fd-an)/max(abs(fd),1e-12)))
out['gradcheck']=[dict(fd=a,adj=b,rel=c) for a,b,c in gc]
for g in gc: print("grad check fd=%.6e adj=%.6e rel=%.2e"%g)

# ---- (4) feasible-perturbation optimality diagnostic + comparator strategies
Jopt=base['J']; Rt=base['Rt']; worse=0; best_pert=np.inf
for trial in range(300):
    d1=np.convolve(rng.normal(size=t.size),np.ones(21)/21,'same'); d2=np.convolve(rng.normal(size=t.size),np.ones(21)/21,'same')
    s=rng.uniform(0.005,0.05)
    a=np.clip(u1+s*d1/np.abs(d1).max(),0,q['umax1']); b=np.clip(u2+s*d2/np.abs(d2).max(),0,q['umax2'])
    over=q['c1']*a+q['c2']*b>Rt
    sc = np.ones_like(Rt)
    sc[over] = Rt[over] / (q['c1']*a[over]+q['c2']*b[over])
    a *= sc
    b *= sc
    Jp=Jof(a,b); best_pert=min(best_pert,Jp); worse+= Jp>=Jopt-1e-6*Jopt
out['perturb']=dict(n=300,n_not_better=int(worse),best_perturbed_J=float(best_pert),J_opt=float(Jopt))
print(out['perturb'])
comp={}
fine=runs[0.25]; tf=fine['t']; Rf=fine['Rt']; zf=np.zeros_like(tf)
def Jf(a,b):
    _,X=rk4_forward(q,a,b,0.25); return objective(tf,X,a,b,q)
comp['no control (0,0)']=Jf(zf,zf)
comp['minimal (0.05,0)']=Jf(zf+0.05,zf)
comp['LLIN at 0.50']=Jf(zf+0.5,zf)
comp['IRS at 0.4167']=Jf(zf,zf+0.5/1.2)
comp['split 0.25/0.208']=Jf(zf+0.25,zf+0.25/1.2)
comp['LLIN = R(t)']=Jf(Rf.copy(),zf)
comp['IRS = R(t)/1.2']=Jf(zf,Rf/1.2)
comp['50/50 capacity split']=Jf(0.5*Rf,0.5*Rf/1.2)
comp['optimal (FBS)']=fine['J']
Jopt_fine=fine['J']
out['comparators']=comp
for k,v in comp.items(): print(f"{k:28s} J={v:.6e}  J/Jopt={v/Jopt_fine:.4f}")

# ---- (5) robustness grid (B1, c2)
rob=[]
for B1 in [5,10,20]:
    for c2 in [0.8,1.2,1.6]:
        qq=dict(q); qq['B1']=B1; qq['c2']=c2; r=fbs(qq,dt=1.0); o=outcomes(r['t'],r['X'])
        rob.append(dict(B1=B1,c2=c2,conv=bool(r['converged']),iters=r['iters'],mean_u1=float(r['u1'].mean()),mean_u2=float(r['u2'].mean()),
                        peak=float(o['peak']),BI=float(o['BI']),max_viol=float((r['use']-r['Rt']).max())))
out['robust']=rob
for r_ in rob: print(r_)
json.dump(out,open('verify_results.json','w'),indent=1,default=float)
np.savez('verify_hist.npz',**{k.replace('=','_').replace('(','').replace(')','').replace(',','_'):v for k,v in hists.items()})
np.savez('verify_profiles.npz',**{f"u1_{dt}":runs[dt]['u1'] for dt in dts},**{f"u2_{dt}":runs[dt]['u2'] for dt in dts},
         **{f"t_{dt}":runs[dt]['t'] for dt in dts},hist_base=base['hist'],Rt=Rt,eta=base['eta'])
