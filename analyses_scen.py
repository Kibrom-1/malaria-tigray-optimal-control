import numpy as np, json, time
from malaria_model import *
from timing import timing_results
from scipy.stats import qmc, rankdata
q=derived(P); out={}; t=np.arange(366.0); Rt=R_cap(t); z=np.zeros_like(t)
def run(a,b,qq=q):
    _,X=rk4_forward(qq,a,b,1.0); o=outcomes(t,X)
    inc=1000*np.trapezoid(qq['alpha_h']*X[:,1],t)/qq['Nh0']          # transitions into infectiousness per 1000
    return X,o,inc
# ---- fixed feasible scenarios (all use <= 0.5 resource units, i.e. feasible for every t)
sc={"No intervention":(0,0),"Minimal reference":(0.05,0),"LLIN only":(0.50,0),"IRS only":(0,0.5/1.2),"LLIN + IRS (split)":(0.25,0.25/1.2)}
ref=None; rows=[]
for k,(a,b) in sc.items():
    X,o,inc=run(z+a,z+b)
    if k=="Minimal reference": ref=o
    rows.append(dict(name=k,u1=a,u2=b,use=a+1.2*b,R0c=R0_parts(q,a,b)[0],peak=o['peak'],t_peak=o['t_peak'],BI=o['BI'],inc=inc))
r=fbs(q,dt=1.0); X,o,inc=run(r['u1'],r['u2'])
rows.append(dict(name="Optimal-control (computed)",u1=float(r['u1'].mean()),u2=float(r['u2'].mean()),use=float(r['use'].mean()),R0c=None,peak=o['peak'],t_peak=o['t_peak'],BI=o['BI'],inc=inc))
for x in rows:
    x['red_peak']=100*(ref['peak']-x['peak'])/ref['peak']; x['red_BI']=100*(ref['BI']-x['BI'])/ref['BI']
    print({k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items()})
out['scenarios']=rows
_, baseline, _ = run(z+0.05,z)
out['timing'] = [dict(name="Minimal reference (no window)", peak=baseline['peak'],
                      BI=baseline['BI'], red_BI=0.0, red_peak=0.0)]
out['timing'] += timing_results(q, x0(q), baseline)
# ---- data-context plausibility: model-implied annual incidence (reference scenario) vs reported 2023/2024
X,o,inc=run(z+0.05,z); out['model_inc_ref']=inc; out['reported_inc']={'2023':50.07,'2024':61.82}
X,o,inc_opt=run(r['u1'],r['u2']); out['model_inc_opt']=inc_opt
print("model-implied transitions into infectiousness per 1000 (reference):",inc," optimal:",inc_opt)
# autonomous-forcing check: phi=0 and seasonality effect on cumulative burden
q0=dict(q); q0['phi']=0.0; _,Xp=rk4_forward(q0,z+0.05,z,1.0,False); out['BI_phi0']=outcomes(t,Xp)['BI']; out['BI_phi045']=outcomes(t,run(z+0.05,z)[0])['BI']
json.dump(out,open('analysis_scen.json','w'),indent=1,default=float)
