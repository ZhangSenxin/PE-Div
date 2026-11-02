"""Grouped fixed-total simulations; raw Gini, no plotting."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy import optimize,special,stats
import capture_model as model
def bins(alpha,B):
    probabilities=np.r_[0.,special.expit(np.linspace(-16,16,B-1)),1.]
    p=np.diff(probabilities)
    boundaries=stats.gamma.ppf(probabilities,a=alpha)
    cdf=special.gammainc(alpha+1,boundaries)
    sf=special.gammaincc(alpha+1,boundaries)
    moments=np.where(probabilities[:-1]<.5,np.diff(cdf),-np.diff(sf))
    means=moments/p
    assert np.isclose(np.dot(p,means),1.,atol=1e-12)
    return p,means

gamma_bins=bins

class Simulator:
    def __init__(self,budget,config,output,keep_replicates=False):
        self.budget=budget;self.seed=int(config['simulation']['seed'])
        self.output=Path(output);self.keep_replicates=keep_replicates
        self.calibration_replicates=int(config['simulation']['lognormal_calibration_replicates'])

    def rng_for(self, *parts):
        key = '|'.join(map(str, (self.seed,) + parts)).encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))

    def lognormal_bins(self, sigma, B):
        probabilities = np.r_[0.0, special.expit(np.linspace(-16, 16, B - 1)), 1.0]
        p = np.diff(probabilities)
        z = special.ndtri(probabilities) - sigma
        moments = np.where(z[:-1] < 0, np.diff(special.ndtr(z)), -np.diff(special.ndtr(-z)))
        y = moments / p
        assert np.all(y >= 0) and np.isclose(np.dot(p, y), 1.0, atol=1e-10)
        return (p, y)

    def weighted_gini(self, counts, abundance, D, mu):
        before = np.cumsum(counts, axis=1) - counts
        return np.sum(counts * abundance * (2 * before + counts - D), axis=1) / (float(D) ** 2 * mu)

    def calibrate_lognormal(self, D, G, B):
        """Finite-D calibration with common multinomial draws, independent validation."""
        mu = self.budget.M / D
        population = math.sqrt(2) * special.ndtri((1 + G / (1 - 1 / mu)) / 2)
        p, _ = self.lognormal_bins(population, B)
        counts = self.rng_for('LN_calibration', D, G, B).multinomial(D, p, size=self.calibration_replicates)

        def objective(sigma):
            _, y = self.lognormal_bins(sigma, B)
            totals = counts @ y
            abundance = 1 + D * (mu - 1) * y[None, :] / totals[:, None]
            return float(self.weighted_gini(counts, abundance, D, mu).mean()) - G
        upper = max(4.0, population * 1.5)
        sigma = optimize.brentq(objective, 0.05, upper, xtol=1e-08)
        return (float(sigma), float(population), float(objective(sigma) + G))

    def draws(self, D, G, B, R, family='dirichlet', sigma=None, tag='main', chunk=256):
        if int(D) != D or D < 1 or R < 2 or (B < 2):
            raise ValueError('Require integer D>=1, R>=2, B>=2')
        D = int(D)
        G = float(G)
        mu = self.budget.M / D
        spec = dict(version=1, D=D, G=G, B=B, R=R, family=family, sigma=sigma, tag=tag, seed=self.seed, M=self.budget.M, beta=self.budget.beta)
        token = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        target = self.output / 'replicates' / f'{family}_G{G:.1f}_D{D}_{token}.npz'
        if self.keep_replicates:
            target.parent.mkdir(exist_ok=True)
        if target.exists():
            with np.load(target) as stored:
                assert json.loads(str(stored['spec'])) == spec
                return {k: stored[k].copy() for k in ('conditional_mean', 'captured', 'gini', 'conditional_variance')}
        rng = self.rng_for(token)
        result = {k: [] for k in ('conditional_mean', 'captured', 'gini', 'conditional_variance')}
        if G == 0:
            probability = self.budget.expected(D, 0) / D
            result = dict(conditional_mean=np.full(R, D * probability), captured=rng.binomial(D, probability, size=R), gini=np.zeros(R), conditional_variance=np.full(R, D * probability * (1 - probability)))
        else:
            if family == 'dirichlet':
                alpha, _ = model.parameters(D, mu, G)
                p, y = gamma_bins(alpha, B)
            elif family == 'lognormal':
                p, y = self.lognormal_bins(sigma, B)
            else:
                raise ValueError(family)
            for begin in range(0, R, chunk):
                counts = rng.multinomial(D, p, size=min(chunk, R - begin))
                totals = counts @ y
                abundance = 1 + D * (mu - 1) * y[None, :] / totals[:, None]
                q = -np.expm1(abundance * np.log1p(-self.budget.beta))
                assert np.isclose(np.sum(counts * abundance, axis=1), self.budget.M, rtol=1e-12).all()
                result['conditional_mean'].append(np.sum(counts * q, axis=1))
                result['captured'].append(rng.binomial(counts, q).sum(axis=1))
                result['gini'].append(self.weighted_gini(counts, abundance, D, mu))
                result['conditional_variance'].append(np.sum(counts * q * (1 - q), axis=1))
            result = {k: np.concatenate(v) for k, v in result.items()}
        assert np.all(result['captured'] <= D) and np.all(result['captured'] >= 0)
        if self.keep_replicates:
            np.savez_compressed(target, spec=json.dumps(spec, sort_keys=True), **result)
        return result

    def summarize(self, D, G, B, R, result, family='dirichlet', sigma=None):
        mean = result['conditional_mean']
        values = result['captured']
        ginis = result['gini']
        reference = self.budget.expected(D, G) if family == 'dirichlet' else float('nan')
        q = np.quantile(values, [0.005, 0.025, 0.5, 0.975, 0.995])
        row = dict(family=family, D_cells=D, target_gini=G, M=self.budget.M, p=self.budget.p, eta=self.budget.eta, beta=self.budget.beta, bins=B if G else 0, replicates=R, sigma=sigma, analytic_D_library=reference, simulated_D_library_mean=float(values.mean()), conditional_mean_average=float(mean.mean()), mean_MC_SE=float(mean.std(ddof=1) / math.sqrt(R)), capture_mean_MC_SE=float(values.std(ddof=1) / math.sqrt(R)), q005=float(q[0]), q025=float(q[1]), median=float(q[2]), q975=float(q[3]), q995=float(q[4]), actual_gini_mean=float(ginis.mean()), actual_gini_SD=float(ginis.std(ddof=1)), gini_mean_MC_SE=float(ginis.std(ddof=1) / math.sqrt(R)), profile_variance=float(mean.var(ddof=1)), conditional_capture_variance_average=float(result['conditional_variance'].mean()), joint_variance=float(values.var(ddof=1)), relative_95_halfwidth=(q[3] - q[1]) / 2 / float(mean.mean()), relative_99_halfwidth=(q[4] - q[0]) / 2 / float(mean.mean()), interval_99_precision='key_position_R_at_least_10000' if R >= 10000 else 'exploratory_R4000')
        if family == 'dirichlet':
            row['mean_error'] = float(mean.mean() - reference)
            row['mean_validation_pass'] = abs(row['mean_error']) <= max(5 * row['mean_MC_SE'], 1e-05 * reference)
        row['gini_validation_pass'] = abs(row['actual_gini_mean'] - G) <= max(5 * row['gini_mean_MC_SE'], 1e-05)
        return row
