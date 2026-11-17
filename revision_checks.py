"""Reproduce threshold, seasonal onset and endpoint checks in the revision."""
import json

import numpy as np
from scipy.optimize import brentq

from malaria_model import P, R0_parts, R_cap, derived, fbs, objective, rk4_forward, x0


def branch_coefficient(parameters):
    _, nonvector, vector_sq = R0_parts(parameters)
    alpha = parameters["alpha_h"]
    mu = parameters["mu_h"]
    gamma = parameters["gamma_h"]
    delta = parameters["delta_h"]
    k_h = alpha + mu
    q_h = gamma + mu + delta
    w = 1/k_h + alpha/(k_h*q_h) + alpha*gamma/(k_h*q_h*(parameters["omega"]+mu))
    kappa = alpha*delta/(k_h*q_h*mu)
    n1 = parameters["a"]*parameters["beta_v"]*alpha/(k_h*q_h)
    return vector_sq*(kappa-w-n1/parameters["mu_v"]) - nonvector*w


def seasonal_onsets(parameters):
    reference = np.full(366, 0.05)
    zero = np.zeros_like(reference)
    initial_state = x0(parameters)
    for _ in range(40):
        _, states = rk4_forward(parameters, reference, zero, x_init=initial_state)
        initial_state = states[-1]
    boundaries = np.insert(np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), 0, 0)
    output = {}
    for name, initial in [("seed", x0(parameters)), ("burn_in", initial_state)]:
        t, states = rk4_forward(parameters, reference, zero, x_init=initial)
        flow = parameters["alpha_h"]*states[:, 1]
        monthly = [np.trapezoid(flow[a:b+1], t[a:b+1]) for a,b in zip(boundaries[:-1], boundaries[1:])]
        output[name] = {"peak_day": int(np.argmax(flow)), "peak_month": int(np.argmax(monthly)+1)}
    return output


def daily_comparisons(parameters):
    candidate = fbs(parameters)
    t = candidate["t"]
    a, b = candidate["u1"], candidate["u2"]
    capacity = R_cap(t)

    def difference(u1, u2):
        assert np.all(u1 + parameters["c2"]*u2 <= capacity+1e-12)
        _, states = rk4_forward(parameters, u1, u2)
        return objective(t, states, u1, u2, parameters)/candidate["J"]-1

    endpoint_a, endpoint_b = a.copy(), b.copy()
    endpoint_a[[0, -1]] = capacity[[0, -1]]
    endpoint_b[[0, -1]] = 0
    return {
        "all_llin_relative_objective_difference": difference(capacity, np.zeros_like(b)),
        "one_percent_move_relative_objective_difference": difference(0.99*a+0.01*capacity, 0.99*b),
        "endpoints_relative_objective_difference": difference(endpoint_a, endpoint_b),
    }


if __name__ == "__main__":
    parameters = derived(P)
    crossing = brentq(lambda a: R0_parts(dict(parameters, a=a))[0]-1, 0.1, 0.3)
    results = {
        "critical_biting_rate": crossing,
        "quadratic_coefficient_at_crossing": branch_coefficient(dict(parameters, a=crossing)),
        "quadratic_coefficient_at_baseline": branch_coefficient(parameters),
        "seasonal_onsets": seasonal_onsets(parameters),
        "daily_comparisons": daily_comparisons(parameters),
    }
    with open("revision_checks.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results, indent=2))
