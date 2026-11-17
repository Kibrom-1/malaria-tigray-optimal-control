import numpy as np, json, time
from scipy.stats import qmc, rankdata
from malaria_model import *
# name -> (low, high). First nine ranges are those stated in the manuscript; the last five are ASSUMED here (see text).
RANGES={ 'a':(0.20,0.30), 'beta_h':(0.32,0.44), 'beta_v':(0.18,0.26), 'mv':(2.0,3.0), 'gamma_h':(0.05,0.15),
 'omega':(0.00137,0.00548), 'pd':(1e-4,1.6e-3), 'eps1':(0.65,0.85), 'eps2':(0.55,0.75), 'phi':(0.25,0.60),
 'tau_h':(0.0005,0.0015), 'mu_v':(0.04,1/15), 'alpha_h':(1/14,1/10), 'alpha_v':(1/12,1/8)}
names=list(RANGES); n=5000
S=qmc.LatinHypercube(d=len(names),seed=42).random(n)
lo=np.array([RANGES[k][0] for k in names]); hi=np.array([RANGES[k][1] for k in names]); Theta=lo+S*(hi-lo)
def prcc(X,y):
    R=np.column_stack([rankdata(X[:,j]) for j in range(X.shape[1])]); ry=rankdata(y); out=[]
    for j in range(X.shape[1]):
        others=np.delete(R,j,axis=1); A=np.column_stack([np.ones(len(ry)),others])
        res_x=R[:,j]-A@np.linalg.lstsq(A,R[:,j],rcond=None)[0]; res_y=ry-A@np.linalg.lstsq(A,ry,rcond=None)[0]
        out.append(np.corrcoef(res_x,res_y)[0,1])
    return np.array(out)
R0v=np.zeros(n); peak=np.zeros(n); BI=np.zeros(n); t=np.arange(366.0); z=np.zeros_like(t); t0=time.time()
for i in range(n):
    p=dict(P); p.update({k:Theta[i,j] for j,k in enumerate(names)}); qq=derived(p)
    R0v[i]=R0_parts(qq,0,0)[0]
    _,X=rk4_forward(qq,z+0.05,z,1.0,True); o=outcomes(t,X); peak[i]=o['peak']; BI[i]=o['BI']
    if i==99: print("est total time",(time.time()-t0)/100*n,"s")
nR=[k for k in names if k not in ('phi','eps2')]   # R0(u=0) does not depend on phi, eps1, eps2; keep eps1 out too
nR=[k for k in names if k not in ('phi','eps1','eps2')]
nT=[k for k in names if k not in ('eps2',)]         # eps2 has no effect when u2=0
pR=prcc(Theta[:,[names.index(k) for k in nR]],R0v)
pT=prcc(Theta[:,[names.index(k) for k in nT]],peak)
pB=prcc(Theta[:,[names.index(k) for k in nT]],BI)
res=dict(names=names,ranges=RANGES,nR=nR,prcc_R0=dict(zip(nR,pR)),nT=nT,prcc_peak=dict(zip(nT,pT)),prcc_BI=dict(zip(nT,pB)),
  R0_quant=np.percentile(R0v,[2.5,50,97.5]).tolist(),peak_quant=np.percentile(peak,[2.5,50,97.5]).tolist(),BI_quant=np.percentile(BI,[2.5,50,97.5]).tolist(),
  frac_R0_gt1=float((R0v>1).mean()))
json.dump(res,open('sens.json','w'),indent=1,default=float)
print(json.dumps(res,indent=1,default=float))
