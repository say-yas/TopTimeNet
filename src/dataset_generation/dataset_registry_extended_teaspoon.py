import traceback

import teaspoon.MakeData.DynSysLib.maps                         as _maps
import teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows as _adf
import teaspoon.MakeData.DynSysLib.driven_dissipative_flows     as _ddf
import teaspoon.MakeData.DynSysLib.conservative_flows           as _cf
import teaspoon.MakeData.DynSysLib.periodic_functions           as _pf
import teaspoon.MakeData.DynSysLib.noise_models                 as _nm
import teaspoon.MakeData.DynSysLib.delayed_flows                as _dlf


# Simulation kwargs per category.
# Total simulated points = L x fs
# flows : L = time in seconds
# maps  : L = number of iterations, fs must be 1 for discrete maps
# noise : same as flows
_KWARGS_BY_CATEGORY = {
    # SampleSize = L * fs; all categories produce exactly 100,000 samples.
    # flow  : L seconds at fs Hz -> 10,000 s x 10 Hz = 100,000 samples
    # map   : fs must = 1 (discrete), so L == SampleSize directly
    # noise : L seconds at fs Hz -> 5,000 s x 20 Hz = 100,000 samples
    'flow':  dict(L=10_000.0, fs=10, SampleSize=100_000),
    'map':   dict(L=100_000,  fs=1,  SampleSize=100_000),
    'noise': dict(L=5_000.0,  fs=20, SampleSize=100_000),
}

# Registry: name -> (function, states, category)
# states = [None] means the function has no dynamic_state parameter.
DATASET_REGISTRY = {
    # Maps
    'logistic_map':                              (_maps.logistic_map,                              ['periodic', 'chaotic'], 'map'),
    'henon_map':                                 (_maps.henon_map,                                 ['periodic', 'chaotic'], 'map'),
    'sine_map':                                  (_maps.sine_map,                                  ['periodic', 'chaotic'], 'map'),
    'tent_map':                                  (_maps.tent_map,                                  ['periodic', 'chaotic'], 'map'),
    'rickers_population_map':                    (_maps.rickers_population_map,                    ['periodic', 'chaotic'], 'map'),
    'gauss_map':                                 (_maps.gauss_map,                                 ['periodic', 'chaotic'], 'map'),
    'pinchers_map':                              (_maps.pinchers_map,                              ['periodic', 'chaotic'], 'map'),
    'sine_circle_map':                           (_maps.sine_circle_map,                           ['periodic', 'chaotic'], 'map'),
    'lozi_map':                                  (_maps.lozi_map,                                  ['periodic', 'chaotic'], 'map'),
    'delayed_logstic_map':                       (_maps.delayed_logstic_map,                       ['periodic', 'chaotic'], 'map'),
    'tinkerbell_map':                            (_maps.tinkerbell_map,                            ['periodic', 'chaotic'], 'map'),
    'burgers_map':                               (_maps.burgers_map,                               ['periodic', 'chaotic'], 'map'),
    'holmes_cubic_map':                          (_maps.holmes_cubic_map,                          ['periodic', 'chaotic'], 'map'),
    'kaplan_yorke_map':                          (_maps.kaplan_yorke_map,                          ['periodic', 'chaotic'], 'map'),
    'gingerbread_man_map':                       (_maps.gingerbread_man_map,                       ['periodic', 'chaotic'], 'map'),
    # Autonomous dissipative flows
    'chua':                                      (_adf.chua,                                       ['periodic', 'chaotic'], 'flow'),
    'lorenz':                                    (_adf.lorenz,                                     ['periodic', 'chaotic'], 'flow'),
    'rossler':                                   (_adf.rossler,                                    ['periodic', 'chaotic'], 'flow'),
    'coupled_lorenz_rossler':                    (_adf.coupled_lorenz_rossler,                     ['periodic', 'chaotic'], 'flow'),
    'coupled_rossler_rossler':                   (_adf.coupled_rossler_rossler,                    ['periodic', 'chaotic'], 'flow'),
    'double_pendulum':                           (_adf.double_pendulum,                            ['periodic', 'chaotic'], 'flow'),
    'diffusionless_lorenz_attractor':            (_adf.diffusionless_lorenz_attractor,             ['periodic', 'chaotic'], 'flow'),
    'complex_butterfly':                         (_adf.complex_butterfly,                          ['periodic', 'chaotic'], 'flow'),
    'chens_system':                              (_adf.chens_system,                               ['periodic', 'chaotic'], 'flow'),
    'hadley_circulation':                        (_adf.hadley_circulation,                         ['periodic', 'chaotic'], 'flow'),
    'ACT_attractor':                             (_adf.ACT_attractor,                              ['periodic', 'chaotic'], 'flow'),
    'linear_feedback_rigid_body_motion_system':  (_adf.linear_feedback_rigid_body_motion_system,   ['periodic', 'chaotic'], 'flow'),
    'moore_spiegel_oscillator':                  (_adf.moore_spiegel_oscillator,                   ['periodic', 'chaotic'], 'flow'),
    'thomas_cyclically_symmetric_attractor':     (_adf.thomas_cyclically_symmetric_attractor,      ['periodic', 'chaotic'], 'flow'),
    'halvorsens_cyclically_symmetric_attractor': (_adf.halvorsens_cyclically_symmetric_attractor,  ['periodic', 'chaotic'], 'flow'),
    'burke_shaw_attractor':                      (_adf.burke_shaw_attractor,                       ['periodic', 'chaotic'], 'flow'),
    'rucklidge_attractor':                       (_adf.rucklidge_attractor,                        ['periodic', 'chaotic'], 'flow'),
    'WINDMI':                                    (_adf.WINDMI,                                     ['periodic', 'chaotic'], 'flow'),
    'simplest_quadratic_chaotic_flow':           (_adf.simplest_quadratic_chaotic_flow,            ['periodic', 'chaotic'], 'flow'),
    'simplest_cubic_chaotic_flow':               (_adf.simplest_cubic_chaotic_flow,                ['periodic', 'chaotic'], 'flow'),
    'simplest_piecewise_linear_chaotic_flow':    (_adf.simplest_piecewise_linear_chaotic_flow,     ['periodic', 'chaotic'], 'flow'),
    'double_scroll':                             (_adf.double_scroll,                              ['periodic', 'chaotic'], 'flow'),
    # Driven dissipative flows
    'driven_pendulum':                           (_ddf.driven_pendulum,                            ['periodic', 'chaotic'], 'flow'),
    'shaw_van_der_pol_oscillator':               (_ddf.shaw_van_der_pol_oscillator,                ['periodic', 'chaotic'], 'flow'),
    'forced_brusselator':                        (_ddf.forced_brusselator,                         ['periodic', 'chaotic'], 'flow'),
    'ueda_oscillator':                           (_ddf.ueda_oscillator,                            ['periodic', 'chaotic'], 'flow'),
    'duffings_two_well_oscillator':              (_ddf.duffings_two_well_oscillator,               ['periodic', 'chaotic'], 'flow'),
    'duffing_van_der_pol_oscillator':            (_ddf.duffing_van_der_pol_oscillator,             ['periodic', 'chaotic'], 'flow'),
    'rayleigh_duffing_oscillator':               (_ddf.rayleigh_duffing_oscillator,                ['periodic', 'chaotic'], 'flow'),
    # Conservative flows
    'simplest_driven_chaotic_flow':              (_cf.simplest_driven_chaotic_flow,                ['periodic', 'chaotic'], 'flow'),
    'nose_hoover_oscillator':                    (_cf.nose_hoover_oscillator,                      ['periodic', 'chaotic'], 'flow'),
    'labyrinth_chaos':                           (_cf.labyrinth_chaos,                             ['chaotic'], 'flow'),
    'henon_heiles_system':                       (_cf.henon_heiles_system,                         ['chaotic'], 'flow'),
    # Periodic functions and noise (no dynamic_state)
    # 'sine':                                      (_pf.sine,                                        [None], 'noise'),
    # 'incommensurate_sine':                       (_pf.incommensurate_sine,                         [None], 'noise'),
    # 'gaussian_noise':                            (_nm.gaussian_noise,                              [None], 'noise'),
    # 'uniform_noise':                             (_nm.uniform_noise,                               [None], 'noise'),
    # 'rayleigh_noise':                            (_nm.rayleigh_noise,                              [None], 'noise'),
    # 'exponential_noise':                         (_nm.exponential_noise,                           [None], 'noise'),
    # Delayed flows
    'mackey_glass':                              (_dlf.mackey_glass,                               ['periodic', 'chaotic'], 'flow'),
}


def load_dataset_long(dataset_name: str, state):
    """
    Load a teaspoon dataset with extended simulation parameters (see
    _KWARGS_BY_CATEGORY): flow and noise categories simulate 100,000
    samples at fs=10 and fs=20 Hz respectively (L=10,000 s and L=5,000 s);
    map categories simulate 100,000 discrete iterations directly (fs=1).

    Returns (t, ts) or None on failure.
    """
    if dataset_name not in DATASET_REGISTRY:
        print(f'  Unknown dataset: {dataset_name}')
        return None

    func, states, category = DATASET_REGISTRY[dataset_name]
    kwargs = _KWARGS_BY_CATEGORY[category]

    if state not in states:
        print(f'  State "{state}" not available for {dataset_name}. '
              f'Available: {states}')
        return None

    try:
        if states == [None]:                         # no dynamic_state param
            return func(**kwargs)
        else:
            return func(dynamic_state=state, **kwargs)
    except Exception as e:
        print(f'  Failed to load {dataset_name} [{state}]: {e}')
        traceback.print_exc()
        return None
