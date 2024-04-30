# MESMER, land-climate dynamics group, S.I. Seneviratne
# Copyright (c) 2021 ETH Zurich, MESMER contributors listed in AUTHORS.
# Licensed under the GNU General Public License v3.0 or later see LICENSE or
# https://www.gnu.org/licenses/
"""
Function to back-transform local variability emulations with MESMER that are standard normal distributions knowing fitted distributions.
"""


import os
import numpy as np
import scipy.stats as ss
from mesmer.utils import read_form_fit_distrib, eval_param_distrib, sigmoid_backtransf





def backtransf_normal2distrib(transf_emus_lv, preds, params_distrib, force_scen=None):
    """Transform inputs over a standard normal distribution into a prescribed fitted distribution.

    Parameters
    ----------
    transf_emus_lv : dict
        nested dictionary for transformed emulators, result of 'create_emus_lv'
        - [targ][scen] (3d array (emus, time, gp) of target for specific scenario)
        
    preds : dict
        nested dictionary with 3 keys: cov_loc, cov_scale, cov_shape. Each one may be empty for no variation of the parameter of the distribution. If not empty, the variables will be used as covariants.
        - [targ][cov_...][covariant][scen]  (1d array (time) of predictor for specific scenario)
        
    params_distrib : dict
        nested dictionary of local variability paramters. Result of the function 'train_l_distrib'.

    force_scen : None or iterable (list, set, 1d array)
        Used to prescribe a specific list of scenarios. If None, they will be deduced from the covariants. Important if no covariants, otherwise the parameters would not depend on scenarios, but using 'force_scen', we can have the desired scenarios.
        
    Returns
    -------
    backtransf_emus
        nested dictionary for back-transformed emulators
        - [targ][scen] (3d array (emus, time, gp) of target for specific scenario)

    Notes
    -----
    - Assumptions:
        The emulators of 'all' are all applied to the same scenarios, the only thing that differs is the distribution of the scenario.
        Implicitly assuming that all scenarios and 'all' have the same length.
    - Disclaimer:
    - TODO:

    """
    # checking that the provided inputs are transformed.
    if list(transf_emus_lv.keys()) != ['all']:
        raise Exception("Data to backtransform must be emulations from 'create_emus_lv' with only the key 'all'")
    
    # creating the dictionary that will be filled in
    backtransf_emus = {var_targ:{} for var_targ in transf_emus_lv['all']}
    
    # looping over all targets to transform
    for var_targ in transf_emus_lv['all'].keys():
        
        # preparing distribution
        distr = params_distrib[var_targ]['distribution']
        if distr in ['gaussian']:
            # this transformation must be performed even if it is from normal to normal, for there are evolution of the parameters.
            distr_ppf = ss.norm.ppf

        elif distr in ['GEV']:
            distr_ppf = ss.genextreme.ppf
            
        elif distr in ['poisson']:
            distr_ppf = ss.poisson.ppf
            # warning, this function is super slow: major source is "scipy.special.pdtrik" used in "scipy.stats.poisson._ppf"
            
        # checking different sizes
        nr_emus, nr_t, nr_gps = transf_emus_lv['all'][var_targ].shape
        
        # preparing positions of points to fill in
        ind_NoNaN = [ii for ii in range(nr_gps) if ii not in params_distrib[var_targ]['gridpoints_nan']]
                
        # Checking the scenarios. Some preds have a list of scenarios, others have 'all', others are meant to be constant.
        list_scens = []
        for inp in preds[var_targ].keys():
            maybe_scens = list( preds[var_targ][inp].keys() )
            maybe_scens.sort()
            if maybe_scens == ['all']:# will be applied to all scens
                pass

            elif len(list_scens)==0:# first time that we find a list of scenarios
                list_scens += maybe_scens # creating this list of scenarios

            elif np.any( list_scens != maybe_scens ):# checking if different scenarios are provided
                raise Exception("The different covariants for the parameters have different list of scenarios, please provide the same ones. NB: 'all' applies to all other scenarios, thus did not cause this issue.")
                
        if force_scen != None:
            list_scens = force_scen

        # evaluate the evolutions of the parameters for these covariants
        params_all = eval_param_distrib( param=params_distrib[var_targ], cov=preds[var_targ], force_scen=list_scens  )
        
        # check for need to do a logistic transformation
        descrip_fit = read_form_fit_distrib( params_distrib[var_targ]['form_fit_distrib'] )
                
        # preparing outputs:
        for scen in list_scens:
            backtransf_emus[var_targ][scen] = np.nan * np.ones( transf_emus_lv['all'][var_targ].shape )
        
        # backtransforming from a normal distribution on the scenarios of the preds
        # objective: one timeserie at a time. It is a good tradeoff speed/RAM.
        for i_emu in np.arange( nr_emus ): # looping on emulators first, because the emulators dont depend on the scenarios for backtransf
            print('backtransforming emulations of', var_targ, ': ', str(np.round(100.*(i_emu+1)/nr_emus,1)), '%', end='\r')
            
            # how likely are the emulated values knowing the standard normal distribution at this year for this scenario?
            p_tmp = ss.norm.cdf( x=transf_emus_lv['all'][var_targ][i_emu,:,ind_NoNaN], loc=0, scale=1).T # can be run once for all scenarios

            for scen in list_scens:
                                
                # to what values of the known GEV would correspond these probabilities?
                if distr in ['GEV']:
                    if params_all[scen]['loc_all'].ndim == 3:# (Emus, Time, GridPoints)
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    c=-params_all[scen]['shape_all'][i_emu,...,ind_NoNaN],
                                                    loc=params_all[scen]['loc_all'][i_emu,...,ind_NoNaN],
                                                    scale=params_all[scen]['scale_all'][i_emu,...,ind_NoNaN]
                                                ).T
                    elif (params_all[scen]['loc_all'].ndim == 2):# (Time, GridPoints): values from params_all have always one axis for time even if constant.
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    c=-params_all[scen]['shape_all'][...,ind_NoNaN],
                                                    loc=params_all[scen]['loc_all'][...,ind_NoNaN],
                                                    scale=params_all[scen]['scale_all'][...,ind_NoNaN]
                                                ).T
                    else:
                        raise Exception('check the dimensions...')

                elif distr in ['gaussian']:
                    if params_all[scen]['loc_all'].ndim == 3:# (Emus, Time, GridPoints)
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    loc=params_all[scen]['loc_all'][i_emu,...,ind_NoNaN],
                                                    scale=params_all[scen]['scale_all'][i_emu,...,ind_NoNaN]
                                                ).T
                    elif (params_all[scen]['loc_all'].ndim == 2):# (Time, GridPoints): values from params_all have always one axis for time even if constant.
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    loc=params_all[scen]['loc_all'][...,ind_NoNaN],
                                                    scale=params_all[scen]['scale_all'][...,ind_NoNaN]
                                                ).T
                    else:
                        raise Exception('check the dimensions...')

                        
                elif distr in ['poisson']:
                    if params_all[scen]['loc_all'].ndim == 3:# (Emus, Time, GridPoints)
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    loc=params_all[scen]['loc_all'][i_emu,...,ind_NoNaN],
                                                    mu=params_all[scen]['mu_all'][i_emu,...,ind_NoNaN]
                                                ).T
                    elif (params_all[scen]['loc_all'].ndim == 2):# (Time, GridPoints): values from params_all have always one axis for time even if constant.
                        data_backPIT = distr_ppf(
                                                    q=p_tmp,
                                                    loc=params_all[scen]['loc_all'][...,ind_NoNaN],
                                                    mu=params_all[scen]['mu_all'][...,ind_NoNaN]
                                                ).T
                    else:
                        raise Exception('check the dimensions...')
                        
                        
                else:
                    raise Exception('Distribution not prepared, please make sure of its type.')
                                
                if 'transfo' in descrip_fit.keys():
                    left = params_distrib[var_targ]['parameters']['transfo_asymptleft'][ind_NoNaN]
                    right = params_distrib[var_targ]['parameters']['transfo_asymptright'][ind_NoNaN]
                    transfo = descrip_fit['transfo']
                    if transfo in ['generalized_logistic', 'generalized_algebraic']:
                        alpha = params_distrib[var_targ]['parameters']['transfo_alpha'][ind_NoNaN]
                        backtransf_emus[var_targ][scen][i_emu,...] = sigmoid_backtransf(
                                                                                            data=data_backPIT,
                                                                                            left=left,
                                                                                            right=right,
                                                                                            type_sigm=transfo,
                                                                                            alpha=alpha
                                                                                            )
                    else:
                        backtransf_emus[var_targ][scen][i_emu,:,ind_NoNaN] = sigmoid_backtransf(
                                                                                            data=data_backPIT,
                                                                                            left=left,
                                                                                            right=right,
                                                                                            type_sigm=transfo
                                                                                            )
                else:
                    backtransf_emus[var_targ][scen][i_emu,:,ind_NoNaN] = data_backPIT
                
        print('')# because of end='\r' in previous print
    return backtransf_emus










