"""Right-tail ISC used for the nominal-alpha selection in the Methods."""
from statistics import NormalDist
import numpy as np


def right_isc(values, alpha, tolerance, max_iterations):
    """Two-sided nominal alpha sets k; right-tail clipping is separate.

    Return the threshold BEFORE updating mean/SD on the stopping iteration,
    matching calculate_isc_alpha.directional_isc used by the final analysis.
    """
    if not 0 < alpha < 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid ISC parameters')
    current = np.asarray(values, dtype=float)
    if not len(current) or not np.isfinite(current).all():
        raise ValueError('ISC requires finite, nonempty scores')
    k = NormalDist().inv_cdf(1 - alpha / 2)
    mu, sd = float(current.mean()), float(current.std(ddof=0))
    history = []
    for iteration in range(1, max_iterations + 1):
        threshold = mu + k * sd
        clipped = current[current <= threshold]
        if not len(clipped):
            raise ValueError('ISC background became empty')
        new_mu, new_sd = float(clipped.mean()), float(clipped.std(ddof=0))
        converged = abs(new_mu - mu) < tolerance and abs(new_sd - sd) < tolerance
        history.append(dict(iteration=iteration, mu=mu, sigma=sd, threshold=threshold,
                            background_n=len(clipped), delta_mu=abs(new_mu - mu), delta_sigma=abs(new_sd - sd)))
        if converged:
            break
        current, mu, sd = clipped, new_mu, new_sd
    return dict(threshold=threshold, k=k, alpha=alpha, iterations=iteration,
                converged=converged, background_mu=mu, background_sigma=sd), history
