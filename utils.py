import numpy as np


def normalization(data, parameters=None):
    """Normalize data in the range [0, 1].
    Args:
        - data: original data, shape (n, d)
        - parameters: if None, default is min/max normalization
    Returns:
        - norm_data: normalized data in [0, 1]
        - norm_parameters: min_val and max_val used for each column, shape (n, d)
    """
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
  
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
        # Return norm_parameters for renormalization
        norm_parameters = {"min_val": min_val, "max_val": max_val}

    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
    
        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
        norm_parameters = parameters
    
    return norm_data, norm_parameters



def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] back to the original range.
    Args:
        - norm_data: normalized data, shape (n, d)
        - norm_parameters: min_val and max_val used for each column
    Returns:
        - renorm_data: renormalized data in the original range, shape (n, d)
    """
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
    return renorm_data



def chen_kipping_2017_radius(planet_massj):
    """Compute planet radius estimate from mass, using Chen &
    Kipping M-R relationship (2017).
    Args:
        - planet_massj: in Jupiter radii
    Return:
        - planet_radj: in Jupiter masses
    """
    RJ = 10.973  # in Earth radii
    MJ = 317.8  # in Earth masses
    planet_masse = planet_massj * MJ
    if planet_masse < 2.04:
        C, S = 0.00346, 0.2790
    elif planet_masse < 132:
        C, S = -0.0925, 0.589
    elif planet_masse < 26600:
        C, S = 1.25, -0.044
    else:
        C, S = -2.85, 0.881
    planet_rade = np.power(10.0, C + np.log10(planet_masse)* S)
    planet_radj = planet_rade / RJ
    return planet_radj



def chen_kipping_2017_mass(planet_radj):
    """Compute planet mass estimate from radius, using Chen &
    Kipping M-R relationship (2017).
    Args:
        - planet_radj: in Jupiter radii
    Return:
        - planet_massh: in Jupiter masses
    """
    RJ = 10.973  # in Earth radii
    MJ = 317.8  # in Earth masses
    planet_rade = planet_radj * RJ
    if planet_rade < 1.23:
        C, S = 0.00346, 0.2790
    elif planet_rade < 11.1:
        C, S = -0.0925, 0.589
    elif planet_rade > 14.3:
        C, S = -2.85, 0.881
    else:
        C, S = np.nan, np.nan
    planet_masse = np.power(10.0, (np.log10(planet_rade) - C) / S)
    planet_massj = planet_masse / MJ
    return planet_massj



def compute_rmse(original_data, miss_data, imputed_data):
    """Compute the RMSE.
    Args:
        - original_data: shape (n, d)
        - miss_data: shape (n, d)
        - imputed_data: shape (n, d)
    Return:
        - rmse: value of the RMSE
    """
    mask_rmse_cells = np.isnan(miss_data) & ~np.isnan(original_data)
    nb_miss = np.sum(mask_rmse_cells)
    squared_diff = (original_data - imputed_data)**2
    sum_of_squares = np.sum(squared_diff[mask_rmse_cells])
    rmse = np.sqrt(sum_of_squares / nb_miss)
    return rmse



def compute_epsilon(imputed_values, true_values):
    """Compute the error epsilon as defined by TLG2020.
    Args:
        - imputed_values: shape (N,) in Jupiter units
        - true_values: shape (N,) in Jupiter units
    Return:
        - eps: Error epsilon
    """
    individual_eps = np.log(true_values) - np.log(imputed_values)
    eps = np.sqrt(np.mean(individual_eps ** 2.0))
    return eps



def msini_convolution_TLG2020(rad_distrib, mass_distrib, true_radii, true_masses, nincl, bool_print=True):
    """Perform the same convolution as TLG2020, for the RV case.
    Args:
        - rad_distrib: in Jupiter radii, shape (N, nb_samples)
        - mass_distrib: in Jupiter masses, shape (N, nb_samples)
        - true_radii: in Jupiter radii, shape (N,)
        - true_masses: in Jupiter masses, shape (N,)
        - nincl: number of convolutions (c.f. TLG2020)
        - bool_print: whether to show progress
    Return:
        - rad_estimates: shape (N, nincl)
        - mass_estimates: shape (N, nincl)
    """
    rad_estimates = np.zeros((mass_distrib.shape[0], nincl))
    mass_estimates = np.zeros((mass_distrib.shape[0], nincl))
    for i1 in range(mass_distrib.shape[0]):
        if bool_print:
            print(f'{i1+1}/{mass_distrib.shape[0]}', end='\r')
        for i2 in range(nincl):
            msini = true_masses[i1] * np.sin(np.arccos(np.random.uniform()))  # creates minimum mass
            mask = (msini < mass_distrib[i1])
            md_temp = mass_distrib[i1, mask]  # mass distrib temp
            rd_temp = rad_distrib[i1, mask]  # rad distrib temp
            iv = np.arcsin(msini / md_temp)  # inclination values
            ipv = (msini ** 2.0 / md_temp ** 3.0) / np.sqrt(1.0 - (msini / md_temp) ** 2.0)  # convolution
            ipv = ipv / np.sum(ipv)
            if np.sum(ipv) > 0.9999:
                rad_estimates[i1, i2] = np.sum(rd_temp * ipv)
                mass_estimates[i1, i2] = np.sum(md_temp * ipv)
            else:  # here should be considered better, maybe...
                print('Problem...')
                rad_estimates[i1, i2] = np.mean(rad_distrib[i1])
                mass_estimates[i1, i2] = msini  
    return rad_estimates, mass_estimates



def convolution_TLG2020_fix_incl(rad_distrib, mass_distrib, true_masses, incl):
    """For the RV case, perform the convolution like TLG2020 to obtain radius and
    mass samples taking into account new weights with minimum mass.
    Args:
        - rad_distrib: in Jupiter radii, shape (N, nb_samples)
        - mass_distrib: in Jupiter masses, shape (N, nb_samples)
        - true_masses: in Jupiter masses, shape (N,)
        - incl: inclination value (in degrees)
    Return:
        - rad_estimates: shape (N,)
        - mass_estimates: shape (N,)
    """
    msini = true_masses * np.sin(incl * np.pi / 2.0 / 90.0)  # generate all minimum masses
    msini = np.expand_dims(msini, axis=-1)
    mask = (mass_distrib >= msini)
    valid_r = np.where(mask, rad_distrib, np.nan)
    valid_m = np.where(mask, mass_distrib, np.nan)
    likelihood = (msini ** 2.0 / valid_m ** 3.0) / np.sqrt(1.0 - (msini / valid_m) ** 2.0)  # convolution
    if np.any(np.sum(mask, axis=-1)==0.0):  # planets with incompatible distrib w.r.t. msini
        idx_pb = np.where(np.sum(mask, axis=-1)==0.0)[0]
        for idx in idx_pb:
            ii = np.argmax(mass_distrib[idx])
            likelihood[idx] = np.zeros(mass_distrib.shape[1])
            likelihood[idx, ii] = 1.0  # just select the largest value in that case
            valid_r[idx, ii] = rad_distrib[idx, ii]
            valid_m[idx, ii] = mass_distrib[idx, ii]
    weights = likelihood / np.expand_dims(np.nansum(likelihood, axis=1), axis=-1)
    rad_estimates = np.nansum(valid_r*weights, axis=1)
    mass_estimates = np.nansum(valid_m*weights, axis=1)
    return rad_estimates, mass_estimates



def convolution_TLG2020_only_msini(rad_distrib, mass_distrib, msini):
    """For the application, perform the convolution like TLG2020 to obtain
    radius and mass samples taking into account new weights with minimum mass.
    The real mass is now unknown!
    Args:
        - rad_distrib: in Jupiter radii, shape (nb_samples,)
        - mass_distrib: in Jupiter masses, shape (nb_samples,)
        - msini: in Jupiter masses, shape ()
    Return:
        - rad_estimates: shape (nb_valid,)
        - mass_estimates: shape (nb_valid,)
        - weights: shape (nb_valid,)
    """
    mask = (mass_distrib >= msini)
    valid_r = np.where(mask, rad_distrib, np.nan)
    valid_m = np.where(mask, mass_distrib, np.nan)
    likelihood = (msini ** 2.0 / valid_m ** 3.0) / np.sqrt(1.0 - (msini / valid_m) ** 2.0)  # convolution
    if np.sum(mask)==0:  # if all mass samples < msini => invalid
        likelihood = np.zeros(mass_distrib.shape[0])
        ii = np.argmax(mass_distrib)
        likelihood[ii] = 1.0  # just select the largest value in that case
        valid_r[ii] = rad_distrib[ii]
        valid_m[ii] = mass_distrib[ii]
    weights = likelihood / np.nansum(likelihood)
    return valid_r, valid_m, weights



# THIS FUNCTION CAN BE REMOVED!!
def importance_sampling(rad_distrib, mass_distrib, true_masses, incl):
    """For the RV case, perform importance sampling to obtain radius and
    mass samples taking into account importance sampling with minimum mass.
    Args:
        - rad_distrib: in Jupiter radii, shape (N, nb_samples)
        - mass_distrib: in Jupiter masses, shape (N, nb_samples)
        - true_masses: in Jupiter masses, shape (N,)
        - incl: inclination value (in degrees)
    Return:
        - rad_estimates: shape (N,)
        - mass_estimates: shape (N,)
    """
    msini = true_masses * np.sin(incl * np.pi / 2.0 / 90.0)  # generate all minimum masses
    mask = (mass_distrib >= np.expand_dims(msini, axis=-1))
    if np.any(np.sum(mask, axis=-1)==0.0):  # planets with incompatible distrib w.r.t. msini
        idx_pb = np.where(np.sum(mask, axis=-1)==0.0)[0]
        for idx in idx_pb:
            ii = np.argmax(mass_distrib[idx])
            mask[idx, ii] = True  # just select the largest value in that case
    valid_r = np.where(mask, rad_distrib, np.nan)
    valid_m = np.where(mask, mass_distrib, np.nan)
    likelihood = np.expand_dims(msini, axis=-1) / valid_m  # sin(arcsin(.))
    weights = likelihood / np.expand_dims(np.nansum(likelihood, axis=1), axis=-1)
    rad_estimates = np.exp(np.nansum(np.log(valid_r)*weights, axis=1))
    mass_estimates = np.exp(np.nansum(np.log(valid_m)*weights, axis=1))
    return rad_estimates, mass_estimates


