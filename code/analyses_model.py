import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from malaria_model import *
q=derived(P); out={}

# ================= A. Autonomous R0 and periodic Rp =================
def R0(q,u1=0,u2=0): return R0_parts(q,u1,u2)
def Rp(q,u1=0.0,u2=0.0):
    """Wang-Zhao periodic reproduction number for seasonal Lambda_v(t) (period 365 d)."""
    dv=q['mu_v']*(1+q['eps2']*u2); a_e=q['a']*(1-q['eps1']*u1)
    kh=q['alpha_h']+q['mu_h']; qh=q['gamma_h']+q['mu_h']+q['delta_h']; kv=q['alpha_v']+dv
    Nh=q['Lam_h']/q['mu_h']
    sol=solve_ivp(lambda t,s: lam_v_rec(t,q)-dv*s, [0,3*365],[q['Lam_v_bar']/dv],dense_output=True,rtol=1e-10,atol=1e-6)
    Sv=lambda t: sol.sol(2*365+(t%365))[0]
    V=np.array([[kh,0,0,0],[-q['alpha_h'],qh,0,0],[0,0,kv,0],[0,0,-q['alpha_v'],dv]])
    def Fm(t):
        m=Sv(t)/Nh
        return np.array([[0,q['tau_h'],0,a_e*q['beta_h']],[0,0,0,0],[0,a_e*q['beta_v']*m,0,0],[0,0,0,0]])
    def rho_of(r):
        f=lambda t,y: ((Fm(t)/r-V)@y.reshape(4,4)).ravel()
        s=solve_ivp(f,[0,365],np.eye(4).ravel(),rtol=1e-9,atol=1e-12)
        return max(abs(np.linalg.eigvals(s.y[:,-1].reshape(4,4))))
    return brentq(lambda r: rho_of(r)-1,0.2,5.0,xtol=1e-8)
out['R0_u0']=R0(q)[0]; out['R0_ref']=R0(q,0.05,0)[0]
out['Rp_u0']=Rp(q); out['Rp_ref']=Rp(q,0.05,0)
# R0 with feasible combined (0.25,0.2083) and LLIN 0.5
out['R0_llin05']=R0(q,0.5,0)[0]; out['R0_irs']=R0(q,0,0.5/1.2)[0]; out['R0_comb']=R0(q,0.25,0.25/1.2)[0]
out['R0_llin_cap']=[R0(q,0.5,0)[0],R0(q,0.8,0)[0]]
print({k:v for k,v in out.items()})

# ================= B. Endemic equilibrium: scalar reduction, uniqueness, bifurcation =================
def eq_pieces(q):
    kh=q['alpha_h']+q['mu_h']; qh=q['gamma_h']+q['mu_h']+q['delta_h']
    c=q['alpha_h']*q['gamma_h']/(kh*qh*(q['omega']+q['mu_h'])); b=1-q['omega']*c
    w=1/kh+q['alpha_h']/(kh*qh)+c
    return kh,qh,c,b,w
def G(lam,q):
    kh,qh,c,b,w=eq_pieces(q)
    nu=q['beta_v']*q['a']*q['alpha_h']*lam/(kh*qh*(1+lam*w))
    mu_h=q['mu_h']; Lh=q['Lam_h']; Lv=q['Lam_v_bar']
    Iv_Nh=q['alpha_v']*nu*Lv*(mu_h+lam*b)/(q['mu_v']*(q['alpha_v']+q['mu_v'])*(nu+q['mu_v'])*Lh*(1+lam*w)) if False else None
    kv=q['alpha_v']+q['mu_v']
    Iv_Nh=q['alpha_v']*nu*Lv*(mu_h+lam*b)/(q['mu_v']*kv*(nu+q['mu_v'])*Lh*(1+lam*w))
    Ih_Nh=q['alpha_h']*lam/(kh*qh*(1+lam*w))
    return q['a']*q['beta_h']*Iv_Nh+q['tau_h']*Ih_Nh, nu, Ih_Nh
def endemic(q):
    f=lambda l: G(l,q)[0]-l
    ls=np.logspace(-12,1,4000); v=np.array([f(l) for l in ls])
    roots=[brentq(f,ls[i],ls[i+1]) for i in range(len(ls)-1) if v[i]*v[i+1]<0]
    return roots
def eq_state(lam,q):
    kh,qh,c,b,w=eq_pieces(q); _,nu,_=G(lam,q)
    Sh=q['Lam_h']/(q['mu_h']+lam*b); Eh=lam*Sh/kh; Ih=q['alpha_h']*Eh/qh; Rh=q['gamma_h']*Ih/(q['omega']+q['mu_h'])
    Sv=q['Lam_v_bar']/(nu+q['mu_v']); Ev=nu*Sv/(q['alpha_v']+q['mu_v']); Iv=q['alpha_v']*Ev/q['mu_v']
    return np.array([Sh,Eh,Ih,Rh,Sv,Ev,Iv])
roots=endemic(q); out['endemic_roots_baseline_u0']=roots
xs=eq_state(roots[0],q); res=rhs(0,xs,0,0,q,seasonal=False)
out['endemic_residual_rel']=float(np.max(np.abs(res)/(np.abs(xs)+1))); out['endemic_state']=xs.tolist()
out['endemic_IhNh']=float(xs[2]/xs[:4].sum()); out['endemic_Nh_over_DFE']=float(xs[:4].sum()/(q['Lam_h']/q['mu_h']))
# Jacobian eigenvalues at endemic equilibrium (finite differences)
def jac(x,q,u1=0,u2=0):
    J=np.zeros((7,7))
    for j in range(7):
        h=1e-6*max(abs(x[j]),1.0); xp=x.copy(); xm=x.copy(); xp[j]+=h; xm[j]-=h
        J[:,j]=(rhs(0,xp,u1,u2,q,False)-rhs(0,xm,u1,u2,q,False))/(2*h)
    return J
ev=np.linalg.eigvals(jac(xs,q)); out['endemic_max_re_eig']=float(ev.real.max())
# condition (C) for uniqueness
kh,qh,c,b,w=eq_pieces(q)
out['condC_lhs']=q['alpha_h']*q['delta_h']/(kh*qh*q['mu_h']); out['condC_w']=w
out['condC_suff_lhs']=q['delta_h']*(q['omega']+q['mu_h']); out['condC_suff_rhs']=q['mu_h']*q['gamma_h']
# check G(lam)/lam strictly decreasing, slope at 0 equals rnv+Rvb2
ls=np.logspace(-10,2,3000); r=np.array([G(l,q)[0]/l for l in ls]); out['G_over_lam_monotone']=bool(np.all(np.diff(r)<0))
_,rnv,Rvb2=R0_parts(q,0,0); out['rho0_check']=[float(r[0]),float(rnv+Rvb2)]
# bifurcation diagram: vary biting rate a
avals=np.linspace(0.08,0.40,161); BD=[]
for av in avals:
    qq=dict(q); qq['a']=av; r0=R0_parts(qq,0,0)[0]
    rts=endemic(qq); lam=max(rts) if rts else 0.0
    ihn=G(lam,qq)[2] if rts else 0.0; BD.append((av,r0,lam,ihn))
BD=np.array(BD); np.save('bifurcation_curve.npy',BD)
# numerical check: long-time autonomous ODE converge to endemic eq for baseline
sol=solve_ivp(lambda t,x: rhs(t,x,0,0,q,False),[0,40*365],x0(q),method='LSODA',rtol=1e-8,atol=1e-6)
out['ode_longrun_IhNh']=float(sol.y[2,-1]/sol.y[:4,-1].sum()); print(out['ode_longrun_IhNh'],out['endemic_IhNh'])
# R0<1 : DFE attraction in simulation
qq=dict(q); qq['a']=0.15; sol2=solve_ivp(lambda t,x: rhs(t,x,0,0,qq,False),[0,20*365],x0(qq),method='LSODA',rtol=1e-8,atol=1e-6)
out['R0_a015']=R0_parts(qq,0,0)[0]; out['Ih_final_a015']=float(sol2.y[2,-1])
# global-stability sufficient condition: m_v replaced by m_v(1+delta_h/mu_h)
mstar=q['mv']*(1+q['delta_h']/q['mu_h']); out['mv_star']=mstar; out['R0_star']=R0_parts(q,0,0,mv=mstar)[0]
# a-threshold values for u: control reproduction numbers
print(json.dumps(out,indent=1,default=float))
json.dump(out,open('analysis_A_B.json','w'),indent=1,default=float)
