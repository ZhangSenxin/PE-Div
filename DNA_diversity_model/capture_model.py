"""Raw-Gini fixed-budget capture model (Methods v1.1)."""
import math
import numpy as np
from scipy import optimize,special

def gini(values):
    ordered=np.sort(values)
    n=len(ordered)
    coeff=2*np.arange(1,n+1,dtype=float)-n-1
    return float(np.sum(coeff*ordered)/(n*np.sum(ordered)))

def gamma_gini(alpha):
    if alpha>1e5:
        return math.exp(-.5*math.log(math.pi*alpha)-1/(8*alpha))
    return math.exp(special.gammaln(alpha+.5)-special.gammaln(alpha+1)-.5*math.log(math.pi))

def parameters(D,mu,target):
    maximum=(1-1/D)*(1-1/mu)
    if not 0<=target<maximum:
        return float('nan'),maximum
    if target==0:
        return float('inf'),maximum
    alpha=math.exp(optimize.brentq(lambda loga:gamma_gini(math.exp(loga))-target/maximum,
                                -25,20,xtol=1e-11))
    return alpha,maximum

def beta_laplace_integral(alpha,b,t):
    """Exact Beta expectation, evaluated by transformed Laguerre quadrature."""
    previous=None
    for order in (32,64,128,256):
        y,weights=special.roots_genlaguerre(order,alpha-1)
        if t>=b:
            # y=t*w resolves sharp capture transitions at high Gini.
            valid=y<t
            logfactor=(b-1)*np.log1p(-y[valid]/t)
            constant=math.exp(special.gammaln(alpha)-special.betaln(alpha,b)-alpha*math.log(t))
            missed=float(constant*np.dot(weights[valid]/weights.sum(),np.exp(logfactor)))
        else:
            w=-np.expm1(-y/b)
            logratio=(alpha-1)*np.log(w/(y/b))
            weights=weights*np.exp(logratio-logratio.max())
            missed=float(np.dot(weights,np.exp(-t*w))/weights.sum())
        if previous is not None and abs(missed-previous)<1e-9:
            return missed
        previous=missed
    raise ArithmeticError(('Beta integral did not converge',alpha,b,t))

def expected_capture(D,mu,target,beta):
    alpha,maximum=parameters(D,mu,target)
    if not math.isfinite(alpha) and target!=0:
        return float('nan'),alpha,maximum
    if target==0:
        return -math.expm1(mu*math.log1p(-beta)),alpha,maximum
    # m_i=1+(M-D)w_i, w~Dirichlet(alpha,...,alpha).
    # The marginal w_i is Beta(alpha,(D-1)*alpha).
    missed=special.hyp1f1(alpha,D*alpha,(mu-1)*D*math.log1p(-beta))
    if not np.isfinite(missed):
        # Substitute y=-(D-1)*alpha*log(1-w) in the exact Beta integral.
        # Generalized Laguerre weights absorb y**(alpha-1)*exp(-y).
        # Divide by the similarly integrated density to avoid cancellation
        # in the normalization constant when Beta parameters are very large.
        b=(D-1)*alpha;t=-(mu-1)*D*math.log1p(-beta)
        missed=beta_laplace_integral(alpha,b,t)
    assert 0<=missed<=1+1e-10,(D,mu,target,alpha,missed)
    capture=1-(1-beta)*missed
    return float(capture),alpha,maximum

class CaptureModel:
    def __init__(self, config):
        p=config['physical']
        self.N0=float(p['initial_cells']);self.p=float(p['editing_efficiency'])
        self.eta=float(p['effective_efficiency']);self.Ns=self.N0*self.p
        self.M=self.Ns*float(p['expansion'])*float(p['copies_per_cell'])
        if self.M.is_integer():self.M=int(self.M)
        self.beta=self.eta*float(p['pre_library_fraction'])
        if not (0<self.p<=1 and 0<self.beta<1 and self.M>=self.Ns>=1):
            raise ValueError('Require 0<p<=1, 0<beta<1 and M>=Ns>=1')

    def expected(self,D,G):
        return float(D)*expected_capture(float(D),self.M/float(D),float(G),self.beta)[0]

    def min_D(self,G):
        if G==0:return 1
        A=(1-G)*self.M+1
        if A*A<4*self.M: return float('inf')
        upper=(A+math.sqrt(A*A-4*self.M))/2
        return int(math.floor(self.M/upper))+1

    def structural_bounds(self,d):
        if not 0<d<=self.Ns:return dict(feasible=False,lower=np.nan,upper=np.nan)
        logmiss=math.log1p(-self.beta)
        equal=lambda D:D*(-math.expm1(logmiss*self.M/D))
        concentrated=lambda D:(D-1)*self.beta-math.expm1(logmiss*(self.M-D+1))
        if d>equal(self.Ns):return dict(feasible=False,lower=np.nan,upper=np.nan)
        lower=d if abs(equal(d)-d)<1e-7 else optimize.brentq(lambda D:equal(D)-d,d,self.Ns,xtol=1e-5)
        upper=self.Ns if concentrated(self.Ns)<=d+1e-7 else optimize.brentq(lambda D:concentrated(D)-d,d,self.Ns,xtol=1e-5)
        return dict(feasible=True,lower=lower,upper=upper)
