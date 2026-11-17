"""Equal-effort timing scenarios for both initial conditions."""
import numpy as np

from malaria_model import R_cap, outcomes, rk4_forward


WINDOWS = {
    "W1: days 1-90": (0, 90),
    "W2: days 91-180": (90, 180),
    "W3: days 181-270": (180, 270),
    "W4: days 276-365": (275, 365),
}


def timing_results(parameters, initial_state, reference):
    base = np.array([0.05, 0.0])
    high = np.array([0.30, 1/6])
    horizon = int(parameters["T"])
    days = np.arange(horizon)
    load_coefficients = np.array([parameters["c1"], parameters["c2"]])
    uniform = base + (high - base) * (90 / horizon)
    target_load = horizon * (uniform @ load_coefficients)
    rows = []

    def add_result(name, controls):
        load = controls @ load_coefficients
        # Capacity increases within each day, so the left endpoints give its minimum.
        assert np.all(load <= R_cap(days) + 1e-12)
        assert np.all(controls >= 0)
        assert np.all(controls <= [parameters["umax1"], parameters["umax2"]])
        assert np.isclose(load.sum(), target_load, rtol=0, atol=1e-10)
        t, states = rk4_forward(parameters, *controls.T, x_init=initial_state,
                                hold_controls=True)
        result = outcomes(t, states)
        rows.append(dict(name=name, peak=result["peak"], BI=result["BI"],
                         red_BI=100*(1-result["BI"]/reference["BI"]),
                         red_peak=100*(1-result["peak"]/reference["peak"])))

    for name, (start, end) in WINDOWS.items():
        active = (days >= start) & (days < end)
        add_result(name, np.where(active[:, None], high, base))
    add_result("Same effort spread over year", np.broadcast_to(uniform, (horizon, 2)))
    return rows
