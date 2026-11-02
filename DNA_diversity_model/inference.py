"""Mean inversion and pointwise 95% conditional-model inverse intervals."""
import math
import numpy as np
import pandas as pd
from scipy import optimize,stats

class Inverter:
    def __init__(self,budget,predictions=None):
        self.model=budget;self.grids={}
        if predictions is not None:
            required={'D_cells','target_gini','q025','q975','analytic_D_library','M','beta'}
            if not required.issubset(predictions):raise ValueError(f'Prediction columns required: {required}')
            p=predictions
            if not np.allclose(p.M,budget.M) or not np.allclose(p.beta,budget.beta):
                raise ValueError('Prediction table and model have different M or beta')
            if 'family' in p:p=p[p.family=='dirichlet']
            for G,g in p.groupby('target_gini'):
                g=g.sort_values('D_cells',kind='stable').drop_duplicates('D_cells',keep='last')
                if (g.q025>g.q975).any():raise ValueError('Reversed prediction limits')
                self.grids[round(float(G),10)]=g

    def quantile(self,D,G,tail):
        m=self.model
        if G==0:
            lo=int(math.floor(D));hi=int(math.ceil(D))
            vals=[stats.binom.ppf(tail,n,m.expected(n,0)/n) for n in (lo,hi)]
            return float(vals[0] if lo==hi else np.interp(D,[lo,hi],vals))
        grid=self.grids[round(float(G),10)];key='q025' if tail==.025 else 'q975'
        # Legacy rule: interpolate quantile offsets, restoring the exact mean.
        # np.interp holds endpoint offsets outside the grid; see README.
        return float(np.clip(m.expected(D,G)+np.interp(D,grid.D_cells,grid[key]-grid.analytic_D_library),0,D))

    def estimate(self,d,G):
        m=self.model;d=float(d);G=float(G)
        if not np.isfinite(d) or d<=0 or not 0<=G<1:raise ValueError('Require finite D_library>0 and 0<=G<1')
        low=max(d,m.min_D(G));top=m.Ns
        row=dict(estimator='jack1',target_gini=G,model_name='Dirichlet',D_library=d,
                 D_cells=np.nan,D_cells_95_cond_lower=np.nan,D_cells_95_cond_upper=np.nan,
                 feasible=False,status='above_model_mean_capacity',interval95_status='not_evaluated',
                 interval_lower_clipped=False,interval_upper_clipped=False,
                 p=m.p,eta=m.eta,beta=m.beta,N_0=m.N0,N_s=m.Ns,M=m.M)
        if low>top:return row
        maximum=m.expected(top,G);row['maximum_expected_D_library']=maximum
        if np.isfinite(maximum) and d<=maximum:
            D=low if abs(m.expected(low,G)-d)<1e-7 else optimize.brentq(lambda D:m.expected(D,G)-d,low,top,xtol=.001)
            row.update(D_cells=D,feasible=True,status='conditional_mean_inverse',forward_residual=m.expected(D,G)-d)
        if G and round(G,10) not in self.grids:
            row['interval95_status']='prediction_grid_not_supplied';return row
        if G and self.grids[round(G,10)].D_cells.max()<top:
            raise ValueError('Prediction grid must cover the cell-count ceiling')
        qhi=lambda D:self.quantile(D,G,.975)
        qlo=lambda D:self.quantile(D,G,.025)
        # Check numerical band ordering/monotonicity before using bracketed roots.
        positions=np.linspace(low,top,65)
        lower=np.array([qlo(D) for D in positions]);upper=np.array([qhi(D) for D in positions])
        if (np.diff(lower)<-1e-7).any() or (np.diff(upper)<-1e-7).any() or (lower>upper).any():
            row['interval95_status']='nonmonotone_or_crossed_bands';return row
        if qhi(top)<d or qlo(low)>d:
            row['interval95_status']='no_feasible_conditional_interval';return row
        lc=qhi(low)>=d;uc=qlo(top)<=d
        row.update(D_cells_95_cond_lower=low if lc else optimize.brentq(lambda D:qhi(D)-d,low,top,xtol=1e-5),
                   D_cells_95_cond_upper=top if uc else optimize.brentq(lambda D:qlo(D)-d,low,top,xtol=1e-5),
                   interval_lower_clipped=bool(lc),interval_upper_clipped=bool(uc),
                   interval95_status='conditional_model_inverse_95')
        return row

def estimate_samples(inputs,budget,ginis,predictions=None):
    frame=inputs.copy()
    if 'estimator' in frame:frame=frame[frame.estimator=='jack1']
    if 'fraction' in frame:frame=frame[np.isclose(frame.fraction,1)]
    if frame.empty or frame.sample_id.duplicated().any():raise ValueError('One full-depth jack1 input per sample is required')
    key='D_library' if 'D_library' in frame else 'D_library_mean'
    inv=Inverter(budget,predictions);rows=[]
    for r in frame.to_dict('records'):
        structural=budget.structural_bounds(float(r[key]))
        for G in ginis:
            row=inv.estimate(r[key],G);row['sample_id']=r['sample_id']
            row.update(D_cells_structure_lower=structural['lower'],D_cells_structure_upper=structural['upper'])
            if 'libraries' in r:row['libraries']=r['libraries']
            rows.append(row)
    detail=pd.DataFrame(rows);summary=[]
    for (est,G),g in detail.groupby(['estimator','target_gini']):
        valid=g[g.feasible];ci=g.dropna(subset=['D_cells_95_cond_lower','D_cells_95_cond_upper'])
        summary.append(dict(estimator=est,target_gini=G,model_name='Dirichlet',
            D_cells_min=valid.D_cells.min(),D_cells_max=valid.D_cells.max(),D_cells_median=valid.D_cells.median(),
            n_unique=len(valid),n_no_solution=len(g)-len(valid),n_total=len(g),
            interval95_envelope_lower=ci.D_cells_95_cond_lower.min(),interval95_envelope_upper=ci.D_cells_95_cond_upper.max(),
            n_interval_compatible=len(ci),p=budget.p,eta=budget.eta,beta=budget.beta,N_s=budget.Ns,M=budget.M))
    return detail,pd.DataFrame(summary)
