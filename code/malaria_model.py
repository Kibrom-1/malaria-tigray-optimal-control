"""
malaria_model.py
Core numerical implementation for:
 "Mathematical Modeling and Optimal Control of Malaria Transmission in
  Conflict-Affected Settings: post-conflict Tigray"
State order: x = [Sh, Eh, Ih, Rh, Sv, Ev, Iv]
Controls   : u1 (LLIN), u2 (IRS);  a_eff = a(1-eps1*u1), mu_v_eff = mu_v(1+eps2*u2)
"""
import numpy as np

# ------------------------------------------------------------------ parameters
P = dict(
    Nh0=6194878.0, mu_h=1/(65*365), alpha_h=1/12, gamma_h=0.10, omega=1/365,
    tau_h=0.001, mu_v=0.05, alpha_v=0.10, a=0.25, beta_h=0.38, beta_v=0.22,
    eps1=0.75, eps2=0.65, umax1=0.88, umax2=0.86, mv=2.5, phi=0.45, sigma=240.0,
    pd=597/2185318, c1=1.0, c2=1.2, A_I=100.0, B1=10.0, B2=8.0, T=365.0)

def derived(p):
    q = dict(p)
    q['delta_h'] = p['pd']*(p['gamma_h']+p['mu_h'])/(1-p['pd'])
    q['Lam_h'] = p['mu_h']*p['Nh0']
    q['Nv0'] = p['mv']*p['Nh0']
    q['Lam_v_bar'] = p['mu_v']*q['Nv0']            # constant (mean) recruitment
    return q

def R_cap(t):                                       # logistical capacity
    return 0.50 + 0.30*np.asarray(t)/365.0

# ------------------------------------------------------------------ dynamics
def lam_v_rec(t, q, seasonal=True):
    if not seasonal:
        return q['Lam_v_bar']
    return q['Lam_v_bar']*(1+q['phi']*np.sin(2*np.pi*(t-q['sigma'])/365.0))

def rhs(t, x, u1, u2, q, seasonal=True):
    Sh,Eh,Ih,Rh,Sv,Ev,Iv = x
    Nh = Sh+Eh+Ih+Rh
    a_e = q['a']*(1-q['eps1']*u1)
    dv  = q['mu_v']*(1+q['eps2']*u2)
    lh = (a_e*q['beta_h']*Iv + q['tau_h']*Ih)/Nh
    lv = a_e*q['beta_v']*Ih/Nh
    return np.array([
        q['Lam_h']+q['omega']*Rh-lh*Sh-q['mu_h']*Sh,
        lh*Sh-(q['alpha_h']+q['mu_h'])*Eh,
        q['alpha_h']*Eh-(q['gamma_h']+q['mu_h']+q['delta_h'])*Ih,
        q['gamma_h']*Ih-(q['omega']+q['mu_h'])*Rh,
        lam_v_rec(t,q,seasonal)-lv*Sv-dv*Sv,
        lv*Sv-(q['alpha_v']+dv)*Ev,
        q['alpha_v']*Ev-dv*Iv])

def x0(q):
    Nh = q['Nh0']; Nv = q['Nv0']
    return np.array([Nh-150, 100, 50, 0, 0.97*Nv, 0.02*Nv, 0.01*Nv], float)

def rk4_forward(q, u1, u2, dt=1.0, seasonal=True, x_init=None, hold_controls=False):
    """u1,u2 arrays on nodes t_k = k*dt (len M+1); controls piecewise-linear in time
       (values at half-steps by averaging), or held constant over each step."""
    M = int(round(q['T']/dt)); t = np.arange(M+1)*dt
    X = np.zeros((M+1,7)); X[0] = x0(q) if x_init is None else x_init
    for k in range(M):
        h = dt; tk = t[k]
        next_control = k if hold_controls else k + 1
        ua1, uc1 = u1[k], u1[next_control]
        ua2, uc2 = u2[k], u2[next_control]
        ub1, ub2 = 0.5*(ua1+uc1), 0.5*(ua2+uc2)
        x = X[k]
        k1 = rhs(tk,x,ua1,ua2,q,seasonal)
        k2 = rhs(tk+h/2,x+h/2*k1,ub1,ub2,q,seasonal)
        k3 = rhs(tk+h/2,x+h/2*k2,ub1,ub2,q,seasonal)
        k4 = rhs(tk+h,x+h*k3,uc1,uc2,q,seasonal)
        X[k+1] = x+h/6*(k1+2*k2+2*k3+k4)
    return t, X

def outcomes(t, X):
    Nh = X[:,:4].sum(1)
    PI = 1000*X[:,2]/Nh
    BI = np.trapezoid(PI, t)
    return dict(peak=PI.max(), t_peak=t[PI.argmax()], BI=BI, PI=PI)

# ------------------------------------------------------------------ adjoint
def adj_rhs_matrix(x, u1, u2, q):
    """Return (dfdx)^T-based adjoint rhs:  psi' = -(df/dx)^T psi - dL/dx.
       Jacobian is analytic via complex-step-free finite closed form."""
    Sh,Eh,Ih,Rh,Sv,Ev,Iv = x
    Nh = Sh+Eh+Ih+Rh
    a_e = q['a']*(1-q['eps1']*u1); dv = q['mu_v']*(1+q['eps2']*u2)
    Ah, Bv = a_e*q['beta_h'], a_e*q['beta_v']
    lh = (Ah*Iv+q['tau_h']*Ih)/Nh; lv = Bv*Ih/Nh
    # derivatives of lh, lv wrt (Sh,Eh,Ih,Rh,Iv)
    dlh = np.array([-lh/Nh,-lh/Nh,(q['tau_h']-lh)/Nh,-lh/Nh,0,0,Ah/Nh])
    dlv = np.array([-lv/Nh,-lv/Nh,(Bv-lv)/Nh,-lv/Nh,0,0,0])
    J = np.zeros((7,7))
    # Sh eq
    J[0,:] = -Sh*dlh; J[0,0] += -lh-q['mu_h']; J[0,3] += q['omega']
    # Eh eq
    J[1,:] = Sh*dlh;  J[1,0] += lh; J[1,1] += -(q['alpha_h']+q['mu_h'])
    # Ih eq
    J[2,1] = q['alpha_h']; J[2,2] = -(q['gamma_h']+q['mu_h']+q['delta_h'])
    # Rh eq
    J[3,2] = q['gamma_h']; J[3,3] = -(q['omega']+q['mu_h'])
    # Sv eq
    J[4,:] = -Sv*dlv; J[4,4] += -lv-dv
    # Ev eq
    J[5,:] = Sv*dlv;  J[5,4] += lv; J[5,5] += -(q['alpha_v']+dv)
    # Iv eq
    J[6,5] = q['alpha_v']; J[6,6] = -dv
    return J

def adj_rhs(t, psi, x, u1, u2, q):
    J = adj_rhs_matrix(x,u1,u2,q)
    dLdx = np.zeros(7); dLdx[2] = q['A_I']
    return -(J.T@psi) - dLdx

def adj_rhs_manuscript(psi, x, u1, u2, q):
    """Closed-form adjoint system exactly as printed in the original manuscript
       (used only to cross-check the Jacobian-based adjoint)."""
    Sh,Eh,Ih,Rh,Sv,Ev,Iv = x; Nh=Sh+Eh+Ih+Rh
    a_e=q['a']*(1-q['eps1']*u1); dv=q['mu_v']*(1+q['eps2']*u2)
    Au,Bu=a_e*q['beta_h'],a_e*q['beta_v']
    lh=(Au*Iv+q['tau_h']*Ih)/Nh; lv=Bu*Ih/Nh
    pS,pE,pI,pR,pSv,pEv,pIv = psi
    Gh=pS-pE; Gv=pSv-pEv
    return np.array([
     Gh*lh*(1-Sh/Nh)-Gv*Sv*lv/Nh+q['mu_h']*pS,
     -Gh*Sh*lh/Nh-Gv*Sv*lv/Nh+(q['alpha_h']+q['mu_h'])*pE-q['alpha_h']*pI,
     -q['A_I']+Gh*Sh/Nh*(q['tau_h']-lh)+Gv*Sv/Nh*(Bu-lv)+(q['gamma_h']+q['mu_h']+q['delta_h'])*pI-q['gamma_h']*pR,
     -Gh*Sh*lh/Nh-Gv*Sv*lv/Nh-q['omega']*pS+(q['omega']+q['mu_h'])*pR,
     Gv*lv+dv*pSv,
     (q['alpha_v']+dv)*pEv-q['alpha_v']*pIv,
     Gh*Au*Sh/Nh+dv*pIv])

def rk4_backward(q, X, u1, u2, dt=1.0):
    M = X.shape[0]-1; t=np.arange(M+1)*dt
    PSI = np.zeros((M+1,7))
    for k in range(M,0,-1):
        h=-dt; x0_,x1_=X[k],X[k-1]; xm=0.5*(x0_+x1_)
        ua1,ub1,uc1=u1[k],0.5*(u1[k]+u1[k-1]),u1[k-1]
        ua2,ub2,uc2=u2[k],0.5*(u2[k]+u2[k-1]),u2[k-1]
        p=PSI[k]
        k1=adj_rhs(t[k],p,x0_,ua1,ua2,q)
        k2=adj_rhs(t[k]+h/2,p+h/2*k1,xm,ub1,ub2,q)
        k3=adj_rhs(t[k]+h/2,p+h/2*k2,xm,ub1,ub2,q)
        k4=adj_rhs(t[k]+h,p+h*k3,x1_,uc1,uc2,q)
        PSI[k-1]=p+h/6*(k1+2*k2+2*k3+k4)
    return PSI

# ------------------------------------------------------------------ KKT control update
def stationary_terms(x, psi, q, u1=0.0, u2=0.0):
    """g1,g2 such that stationarity reads  B_i u_i + g_i + c_i eta = 0.
       (g1 depends on u only through nothing; g2 neither: both are linear-in-u free.)"""
    Sh,Eh,Ih,Rh,Sv,Ev,Iv = x; Nh=Sh+Eh+Ih+Rh
    g1 = q['eps1']*q['a']/Nh*(q['beta_h']*Sh*Iv*(psi[0]-psi[1])
                              +q['beta_v']*Sv*Ih*(psi[4]-psi[5]))
    g2 = -q['mu_v']*q['eps2']*(Sv*psi[4]+Ev*psi[5]+Iv*psi[6])
    return g1, g2

def kkt_update(g1, g2, Rt, q):
    """Pointwise solve of  min_u  B1/2 u1^2 + B2/2 u2^2 + g1 u1 + g2 u2
       s.t. 0<=u1<=umax1, 0<=u2<=umax2, c1 u1 + c2 u2 <= R(t)
       -> KKT with multiplier eta>=0 (bisection on eta)."""
    c1,c2,B1,B2=q['c1'],q['c2'],q['B1'],q['B2']
    def u_of(eta):
        return (np.clip(-(g1+c1*eta)/B1,0,q['umax1']),
                np.clip(-(g2+c2*eta)/B2,0,q['umax2']))
    u1,u2=u_of(0.0)
    if c1*u1+c2*u2<=Rt+1e-14: return u1,u2,0.0
    lo,hi=0.0,1.0
    while c1*u_of(hi)[0]+c2*u_of(hi)[1]>Rt:
        hi*=2
        if hi>1e12: break
    for _ in range(100):
        mid=0.5*(lo+hi); a,b=u_of(mid)
        if c1*a+c2*b>Rt: lo=mid
        else: hi=mid
    eta=hi; u1,u2=u_of(eta)
    return u1,u2,eta


def kkt_update_vec(g1, g2, Rt, q):
    """Vectorised pointwise KKT solve over all time nodes (bisection on eta)."""
    c1,c2,B1,B2=q['c1'],q['c2'],q['B1'],q['B2']
    def u_of(eta):
        return (np.clip(-(g1+c1*eta)/B1,0,q['umax1']),
                np.clip(-(g2+c2*eta)/B2,0,q['umax2']))
    u1,u2=u_of(0.0*g1)
    act=c1*u1+c2*u2>Rt+1e-14
    eta=np.zeros_like(g1)
    if act.any():
        lo=np.zeros_like(g1); hi=np.ones_like(g1)
        for _ in range(60):                      # bracket
            a,b=u_of(hi)
            m=(c1*a+c2*b>Rt)&act
            if not m.any(): break
            hi=np.where(m,hi*2,hi)
        for _ in range(80):
            mid=0.5*(lo+hi); a,b=u_of(mid)
            over=c1*a+c2*b>Rt
            lo=np.where(over,mid,lo); hi=np.where(over,hi,mid)
        eta=np.where(act,hi,0.0)
        u1,u2=u_of(eta)
    return u1,u2,eta

def stationary_terms_vec(X, PSI, q):
    Sh,Eh,Ih,Rh,Sv,Ev,Iv = X.T; Nh=Sh+Eh+Ih+Rh
    g1 = q['eps1']*q['a']/Nh*(q['beta_h']*Sh*Iv*(PSI[:,0]-PSI[:,1])
                              +q['beta_v']*Sv*Ih*(PSI[:,4]-PSI[:,5]))
    g2 = -q['mu_v']*q['eps2']*(Sv*PSI[:,4]+Ev*PSI[:,5]+Iv*PSI[:,6])
    return g1,g2

def objective(t, X, u1, u2, q):
    Ih=X[:,2]
    return np.trapezoid(q['A_I']*Ih+0.5*q['B1']*u1**2+0.5*q['B2']*u2**2, t)

# ------------------------------------------------------------------ forward-backward sweep
def fbs(q, dt=1.0, theta=0.15, tol=1e-6, maxit=150, u_init=(0.05,0.0), seasonal=True,
        verbose=False, cap=R_cap, return_hist=True):
    M=int(round(q['T']/dt)); assert abs(M*dt-q['T'])<1e-9, 'dt must divide T'
    t=np.arange(M+1)*dt; Rt=cap(t)
    u1=np.full(M+1,u_init[0],float); u2=np.full(M+1,u_init[1],float)
    # make initial guess feasible
    for k in range(M+1):
        if q['c1']*u1[k]+q['c2']*u2[k]>Rt[k]:
            s=Rt[k]/(q['c1']*u1[k]+q['c2']*u2[k]); u1[k]*=s; u2[k]*=s
    hist=[]; converged=False
    for it in range(1,maxit+1):
        _,X=rk4_forward(q,u1,u2,dt,seasonal)
        PSI=rk4_backward(q,X,u1,u2,dt)
        g1,g2=stationary_terms_vec(X,PSI,q)
        n1,n2,eta=kkt_update_vec(g1,g2,Rt,q)
        u1n=theta*n1+(1-theta)*u1; u2n=theta*n2+(1-theta)*u2
        delta=max(np.abs(u1n-u1).max(),np.abs(u2n-u2).max())
        hist.append(delta); u1,u2=u1n,u2n
        if verbose and it%10==0: print(it,delta)
        if delta<tol: converged=True; break
    _,X=rk4_forward(q,u1,u2,dt,seasonal)
    PSI=rk4_backward(q,X,u1,u2,dt)
    J=objective(t,X,u1,u2,q)
    res=dict(t=t,X=X,PSI=PSI,u1=u1,u2=u2,eta=eta,J=J,iters=it,converged=converged,
             hist=np.array(hist),Rt=Rt,use=q['c1']*u1+q['c2']*u2)
    return res

# ------------------------------------------------------------------ analytical threshold
def R0_parts(q, u1=0.0, u2=0.0, mv=None):
    """Autonomous basic reproduction number (constant Lambda_v)."""
    mv=q['mv'] if mv is None else mv
    mv=mv/(1+q['eps2']*u2)          # disease-free vector density under IRS: Lambda_v/(mu_v(1+eps2 u2))
    kh=q['alpha_h']+q['mu_h']; qh=q['gamma_h']+q['mu_h']+q['delta_h']
    dv=q['mu_v']*(1+q['eps2']*u2); kv=q['alpha_v']+dv
    a_e=q['a']*(1-q['eps1']*u1)
    rnv=q['tau_h']*q['alpha_h']/(kh*qh)
    Rvb2=a_e**2*q['beta_h']*q['beta_v']*q['alpha_h']*q['alpha_v']*mv/(kh*qh*kv*dv)
    R0=0.5*(rnv+np.sqrt(rnv**2+4*Rvb2))
    return R0,rnv,Rvb2

# ------------------------------------------------------------------ periodic reproduction number
def Rp_periodic(q, u1=0.0, u2=0.0):
    """Reproduction number R_p of the periodically forced model (Wang & Zhao, 2008; Bacaer & Guernaoui, 2006).
       R_p is the value rho for which the monodromy matrix of x' = (F(t)/rho - V) x has spectral radius 1."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    dv=q['mu_v']*(1+q['eps2']*u2); a_e=q['a']*(1-q['eps1']*u1)
    kh=q['alpha_h']+q['mu_h']; qh=q['gamma_h']+q['mu_h']+q['delta_h']; kv=q['alpha_v']+dv
    Nh=q['Lam_h']/q['mu_h']
    sol=solve_ivp(lambda t,s: lam_v_rec(t,q)-dv*s,[0,3*365],[q['Lam_v_bar']/dv],dense_output=True,rtol=1e-10,atol=1e-6)
    Sv=lambda t: sol.sol(2*365+(t%365))[0]            # periodic disease-free vector profile
    V=np.array([[kh,0,0,0],[-q['alpha_h'],qh,0,0],[0,0,kv,0],[0,0,-q['alpha_v'],dv]])
    def Fm(t):
        m=Sv(t)/Nh
        return np.array([[0,q['tau_h'],0,a_e*q['beta_h']],[0,0,0,0],[0,a_e*q['beta_v']*m,0,0],[0,0,0,0]])
    def rho_of(r):
        f=lambda t,y: ((Fm(t)/r-V)@y.reshape(4,4)).ravel()
        s=solve_ivp(f,[0,365],np.eye(4).ravel(),rtol=1e-9,atol=1e-12)
        return max(abs(np.linalg.eigvals(s.y[:,-1].reshape(4,4))))
    return brentq(lambda r: rho_of(r)-1,0.1,6.0,xtol=1e-8)
