import json, numpy as np
from malaria_model import *
q=derived(P); R0s=[]; Rps=[]
for a in np.linspace(0.10,0.36,14):
    qq=dict(q); qq['a']=a; R0s.append(R0_parts(qq,0,0)[0]); Rps.append(Rp_periodic(qq,0,0))
json.dump(dict(R0=R0s,Rp=Rps),open('rp_curve.json','w'))
print(np.round(R0s,3)); print(np.round(Rps,3)); print("max rel. diff",np.max(np.abs(np.array(Rps)/np.array(R0s)-1)))
# also check Rp for phi values at baseline
for ph in [0.0,0.25,0.45,0.60]:
    qq=dict(q); qq['phi']=ph; print("phi",ph,"Rp",round(Rp_periodic(qq,0,0),4))
