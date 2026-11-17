import numpy as np, json
from malaria_model import *
from timing import timing_results
q=derived(P); t=np.arange(366.0); z=np.zeros_like(t); Rt=R_cap(t)
# burn-in 40 years under the minimal reference to reach quasi-periodic state at 1 January
x=x0(q); drift=[]
for yr in range(40):
    _,X=rk4_forward(q,z+0.05,z,1.0,True,x_init=x); x_new=X[-1]; drift.append(np.max(np.abs(x_new-x)/np.maximum(np.abs(x), 1.0))); x=x_new
print("year-to-year relative drift at end of burn-in:",drift[-1])
xb=x.copy(); np.save('burnin_state.npy',xb)
def run(a,b):
    _,X=rk4_forward(q,a,b,1.0,True,x_init=xb); o=outcomes(t,X); inc=1000*np.trapezoid(q['alpha_h']*X[:,1],t)/q['Nh0']; return X,o,inc
_,o0,inc0=run(z+0.05,z)
out={'ref':dict(peak=o0['peak'],t_peak=o0['t_peak'],BI=o0['BI'],inc=inc0)}
rows = timing_results(q, xb, o0)
# prevalence seasonal profile in burn-in year
Xr,_,_=run(z+0.05,z); PI=1000*Xr[:,2]/Xr[:,:4].sum(1); out['PI_min_day']=int(PI.argmin()); out['PI_max_day']=int(PI.argmax()); out['PI_range']=[float(PI.min()),float(PI.max())]
print(out['PI_min_day'],out['PI_max_day'],out['PI_range'])
out['rows']=rows; out['drift']=float(drift[-1]); out['inc_ref']=inc0
json.dump(out,open('analysis_burnin.json','w'),indent=1,default=float)
