import numpy as np

BASE_ANNUAL = 1000.0 # illustrative base premium USD

# Score expected in [0, 1]; convert to 0-100 display elsewhere

def behavior_multiplier(score01: float) -> float:
    # Linear map with caps; lower score => cheaper
    mult = 0.75 + 0.5*(1.0 - score01)
    return float(np.clip(mult, 0.65, 1.25))


def quote(score01: float, base: float = BASE_ANNUAL) -> float:
    mult = behavior_multiplier(score01)
    premium = base * mult
    return {'base': base, 'multiplier': round(mult, 3), 'annual_premium': round(premium, 2)}



