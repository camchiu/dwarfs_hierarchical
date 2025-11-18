# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pymc
import arviz
import corner
from scipy.stats import chi2
import pytensor.tensor as pt

# velocity distribution -- Gaussian MCMC
def velocity_mcmc(vel, vel_err, vr_guess, sig_guess, draws = 5000, tune = 2000):
    with pymc.Model() as model:
        # set priors
        v_r = pymc.Uniform("v_r", lower = -400, upper = 400, initval = vr_guess)
        log_sigma_v = pymc.Uniform("log_sigma_v", lower = np.log(0.1), upper = np.log(20), initval = np.log(sig_guess))
        sigma_v = pymc.Deterministic("sigma_v", pymc.math.exp(log_sigma_v))

        # set likelihood
        total_err = np.sqrt(vel_err**2 + sigma_v**2)
        pymc.Normal("obs", mu = v_r, sigma = total_err, observed = vel)

        # sample
        trace = pymc.sample(draws = draws, tune = tune, chains = 4, target_accept = 0.9, 
                            return_inferencedata = True, progressbar = True)

    # get summary and plot posterior
    summary = arviz.summary(trace, var_names=["v_r", "sigma_v"], hdi_prob = 0.68)
    arviz.plot_posterior(trace, var_names=["v_r", "sigma_v"], hdi_prob = 0.68)

    # plot corner
    sample_data = arviz.extract_dataset(trace, group = "posterior", var_names = ["v_r", "sigma_v"]).to_dataframe()
    fig = corner.corner(sample_data, labels = ["$v_r$", "$\sigma_v$"], quantiles = [0.16, 0.5, 0.84], 
                        show_titles = True, title_kwargs = {"fontsize": 12})
    plt.show()

    # get output
    posterior = arviz.extract_dataset(trace, group = "posterior", var_names = ["v_r", "sigma_v"]).to_dataframe()
    vr_vals = np.percentile(posterior["v_r"], [16, 50, 84])
    sigma_vals = np.percentile(posterior["sigma_v"], [16, 50, 84])

    return model, trace, summary, fig, vr_vals, sigma_vals

# metallicity distribution -- Gaussian MCMC
def feh_mcmc(feh, feh_err, vr_guess, sig_guess, draws = 5000, tune = 2000):
    with pymc.Model() as model:
        # set priors
        feh_mean = pymc.Uniform("feh_mean", lower = -6, upper = 2, initval = vr_guess)
        log_sigma_feh = pymc.Uniform("log_sigma_feh", lower = np.log(0.01), upper = np.log(4), initval = np.log(sig_guess))
        sigma_feh = pymc.Deterministic("sigma_feh", pymc.math.exp(log_sigma_feh))

        # set likelihood
        total_err = np.sqrt(feh_err**2 + sigma_feh**2)
        pymc.Normal("obs", mu = feh_mean, sigma = total_err, observed = feh)

        # sample
        trace = pymc.sample(draws = draws, tune = tune, chains = 4, target_accept = 0.9, 
                            return_inferencedata = True, progressbar = True)

    # get summary and plot posterior
    summary = arviz.summary(trace, var_names=["feh_mean", "sigma_feh"], hdi_prob = 0.68)
    arviz.plot_posterior(trace, var_names=["feh_mean", "sigma_feh"], hdi_prob = 0.68)

    # plot corner
    sample_data = arviz.extract_dataset(trace, group = "posterior", var_names = ["feh_mean", "sigma_feh"]).to_dataframe()
    fig = corner.corner(sample_data, labels = ["[Fe/H]", r"$\sigma_{\rm [Fe/H]}$"], quantiles = [0.16, 0.5, 0.84], 
                        show_titles = True, title_kwargs = {"fontsize": 12})
    plt.show()

    # get output
    posterior = arviz.extract_dataset(trace, group = "posterior", var_names = ["feh_mean", "sigma_feh"]).to_dataframe()
    feh_vals = np.percentile(posterior["feh_mean"], [16, 50, 84])
    sigma_vals = np.percentile(posterior["sigma_feh"], [16, 50, 84])

    return model, trace, summary, fig, feh_vals, sigma_vals

# 2 - population level hyperparameters
def velocity_mcmc2(vels, v_errs, galaxy_ids, draws = 4000, tune = 2000):
    n_galaxies = np.unique(galaxy_ids).size

    with pymc.Model() as model:
        # 1) Population hyperparameters
        mu_vr = pymc.Normal("mu_vr", mu = 0, sigma = 10)                     # mean of global systemic velocity distribution
        log_tau_vr = pymc.Uniform("log_tau_vr", lower = np.log(50), upper = np.log(200))
        tau_vr = pymc.Deterministic("tau_vr", pymc.math.exp(log_tau_vr))     # population scatter in global systemic velocity distribution

        mu_sig = pymc.TruncatedNormal("mu_sig", mu = 6, sigma = 2, lower = 0, upper = 15)    # mean of velocity dispersion distribution
        log_tau_sig = pymc.Uniform("log_tau_sig", lower = np.log(1), upper = np.log(10))   # population scatter in velocity dispersion distribution
        tau_sig = pymc.Deterministic("tau_sig", pymc.math.exp(log_tau_sig))

        # 2) Galaxy parameters
        v_r = pymc.Normal("v_r", mu = mu_vr, sigma = tau_vr, shape = n_galaxies)
        sigma_v = pymc.TruncatedNormal('sigma_v', mu = mu_sig, sigma = tau_sig, lower = 0, upper = 20, shape = n_galaxies)

        # likelihood
        total_verr = np.sqrt(v_errs**2 + sigma_v[galaxy_ids]**2)
        pymc.Normal("obs", mu = v_r[galaxy_ids], sigma = total_verr, observed = vels)

        # sampling
        trace = pymc.sample(draws = draws, tune = tune, chains = 4, target_accept = 0.9, 
                            return_inferencedata = True, progressbar = True)

    # population
    summary = arviz.summary(trace, var_names = ["mu_vr", "tau_vr", "mu_sig", "tau_sig", "v_r", "sigma_v"], hdi_prob = 0.68)
    posterior = trace.posterior
    mu_vr_vals = np.percentile(posterior["mu_vr"].values.flatten(), [16, 50, 84])
    tau_vr_vals = np.percentile(posterior["tau_vr"].values.flatten(), [16, 50, 84])
    mu_sig_vals = np.percentile(posterior["mu_sig"].values.flatten(), [16, 50, 84])
    tau_sig_vals = np.percentile(posterior["tau_sig"].values.flatten(), [16, 50, 84])

    # galaxies
    v_r_samples = posterior["v_r"].values
    sigma_v_samples = posterior["sigma_v"].values

    n_galaxies = v_r_samples.shape[2]
    v_r_percentiles = np.zeros((n_galaxies, 3)) 
    sigma_v_percentiles = np.zeros((n_galaxies, 3))

    for ii in range(n_galaxies):
        v_r_percentiles[ii] = np.percentile(v_r_samples[:, :, ii].flatten(), [16, 50, 84])
        sigma_v_percentiles[ii] = np.percentile(sigma_v_samples[:, :, ii].flatten(), [16, 50, 84])

    return model, trace, summary, mu_vr_vals, tau_vr_vals, mu_sig_vals, tau_sig_vals, v_r_percentiles, sigma_v_percentiles

def velocity_mwf(vels, v_errs, galaxy_ids, draws = 4000, tune = 2000, mw_mu = 0, mw_sigma = 120):   # Milky Way distribution parameters
    n_galaxies = np.size(np.unique(galaxy_ids))

    with pymc.Model() as model:
        # 1) Population hyperparameters
        mu_vr = pymc.Normal("mu_vr", mu = 0, sigma = 10)                     # mean of global systemic velocity distribution
        log_tau_vr = pymc.Uniform("log_tau_vr", lower = np.log(50), upper = np.log(200))
        tau_vr = pymc.Deterministic("tau_vr", pymc.math.exp(log_tau_vr))     # population scatter in global systemic velocity distribution

        mu_sig = pymc.TruncatedNormal("mu_sig", mu = 6, sigma = 2, lower = 0, upper = 15)    # mean of velocity dispersion distribution
        log_tau_sig = pymc.Uniform("log_tau_sig", lower = np.log(1), upper = np.log(10))   # population scatter in velocity dispersion distribution
        tau_sig = pymc.Deterministic("tau_sig", pymc.math.exp(log_tau_sig))

        # 2) Galaxy parameters
        v_r = pymc.Normal("v_r", mu = mu_vr, sigma = tau_vr, shape = n_galaxies)
        sigma_v = pymc.TruncatedNormal('sigma_v', mu = mu_sig, sigma = tau_sig, lower = 0, upper = 20, shape = n_galaxies)
        
        # 3) Milky Way contamination fraction
        f_MW = pymc.Beta("f_MW", alpha = 1, beta = 9)  

        # galaxy
        total_verr = pymc.math.sqrt(v_errs**2 + sigma_v[galaxy_ids]**2)
        galaxy_comp = pymc.Normal.dist(mu = v_r[galaxy_ids], sigma = total_verr)

        # MW foreground
        mw_comp = pymc.Normal.dist(mu = mw_mu, sigma = mw_sigma)

        # likelihood -- Gaussian mixture
        pymc.Mixture("obs", w = [1 - f_MW, f_MW], comp_dists = [galaxy_comp, mw_comp], observed = vels)

        # sampling
        trace = pymc.sample(draws = draws, tune = tune, chains = 1, target_accept = 0.9, return_inferencedata = True)

    # summary
    posterior = trace.posterior
    mu_vr_vals = np.percentile(posterior["mu_vr"].values.flatten(), [16, 50, 84])
    tau_vr_vals = np.percentile(posterior["tau_vr"].values.flatten(), [16, 50, 84])
    mu_sig_vals = np.percentile(posterior["mu_sig"].values.flatten(), [16, 50, 84])
    tau_sig_vals = np.percentile(posterior["tau_sig"].values.flatten(), [16, 50, 84])

    v_r_samples = posterior["v_r"].values
    sigma_v_samples = posterior["sigma_v"].values

    n_galaxies = v_r_samples.shape[2]
    v_r_percentiles = np.zeros((n_galaxies, 3))
    sigma_v_percentiles = np.zeros((n_galaxies, 3))

    for ii in range(n_galaxies):
        v_r_percentiles[ii] = np.percentile(v_r_samples[:, :, ii].flatten(), [16, 50, 84])
        sigma_v_percentiles[ii] = np.percentile(sigma_v_samples[:, :, ii].flatten(), [16, 50, 84])

    return (model, trace, mu_vr_vals, tau_vr_vals, mu_sig_vals, tau_sig_vals, v_r_percentiles, sigma_v_percentiles)

# full bayesian hierarchial model with binary
def velocity_binary(vels, v_errs, galaxy_ids, star_ids, draws = 4000, tune = 2000,
                    lam_prior = 0.7, sigma_bin_prior = 20, 
                    return_pvals = True, chains = 4,target_accept = 0.9):
    """
    Hierarchical multi-epoch velocity model with per-galaxy binary fractions and per-galaxy binary jitter amplitude.
    
    Parameters:
        vels, v_errs: 1D array-like of observed velocities and errors
        galaxy_ids: galaxy index per observation
        star_ids: star index per observation
        lam_prior: weight of prior on per-star variability
        sigma_bin_prior: scale of HalfNormal prior for binary jitter
        map_w_prior: method to convert p-values to soft prior ('1-p' or 'logistic')
        logistic_a: logistic scale for 'logistic' mapping
        return_pvals: whether to include p-values and chi2 in summaries
        chains, target_accept: PyMC sampling options
    """
    
    # sanitize array to work with pymc
    def to_array(arr, default_value = 0.1):
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(default_value)
        arr = np.asarray(arr, dtype = float)
        arr = np.nan_to_num(arr, nan = default_value, posinf = default_value, neginf = default_value)
        return arr

    # convert inputs to clean numpy arrays
    vels = to_array(vels)
    v_errs = to_array(v_errs)
    galaxy_ids = np.asarray(galaxy_ids, dtype = int)
    star_ids = np.asarray(star_ids, dtype = int)

    # map galaxy/star indicies to integer indicies
    unique_gals, gal_inverse = np.unique(galaxy_ids, return_inverse = True)
    n_galaxies = len(unique_gals)

    combos = np.vstack([galaxy_ids, star_ids]).T
    unique_stars, star_inverse = np.unique(combos, axis = 0, return_inverse = True)
    n_stars = len(unique_stars)

    # galaxy of each star
    gal_of_star = np.array([gal_inverse[np.where(star_inverse==s)[0][0]] for s in range(n_stars)])

    # compute chi2, pvalues
    p_vals = np.ones(n_stars)
    chi2_vals = np.zeros(n_stars)
    n_epochs = np.zeros(n_stars, dtype=int)

    for s in range(n_stars):
        idx = np.where(star_inverse == s)[0]
        n = len(idx)
        n_epochs[s] = n
        if n <= 1: # only one epoch
            p_vals[s] = 1.0
            chi2_vals[s] = 0.0
            continue
        vi = vels[idx]
        si = v_errs[idx]
        w = 1.0 / (si**2)
        vbar = np.sum(w*vi)/np.sum(w)
        chi2_s = np.sum(((vi - vbar)**2)/si**2)
        dof = n-1
        p = chi2.sf(chi2_s, dof) if dof > 0 else 1.0
        if np.isnan(p) or np.isinf(p):
            p = 0.1
        p = np.clip(p, 1e-300, 1.0)
        p_vals[s] = p
        chi2_vals[s] = chi2_s

    # convert p-values to soft prior weight
    w_prior_star = 1.0 - p_vals
    w_prior_star = np.nan_to_num(w_prior_star, nan=0.1, posinf=0.1, neginf=0.1)
    w_prior_star = np.clip(w_prior_star, 0.0, 1.0)

    # hierarchical model
    with pymc.Model() as model:
        # 1) Population hyperparameters
        mu_vr = pymc.Normal("mu_vr", mu = 0, sigma = 10)
        log_tau_vr = pymc.Uniform("log_tau_vr", lower = np.log(50), upper = np.log(200))
        tau_vr = pymc.Deterministic("tau_vr", pymc.math.exp(log_tau_vr))

        mu_sig = pymc.TruncatedNormal("mu_sig", mu = 6, sigma = 2, lower = 0, upper = 15)
        log_tau_sig = pymc.Uniform("log_tau_sig", lower = np.log(1), upper = np.log(10))
        tau_sig = pymc.Deterministic("tau_sig", pymc.math.exp(log_tau_sig))

        # 2) Galaxy-level parameters
        v_r = pymc.Normal("v_r", mu = mu_vr, sigma = tau_vr, shape = n_galaxies)
        sigma_v = pymc.TruncatedNormal("sigma_v", mu = mu_sig, sigma = tau_sig, lower = 0, upper = 100, shape = n_galaxies)

        # 3) Per-galaxy binary fraction and jitter
        f_bin = pymc.Beta("f_bin", alpha = np.ones(n_galaxies), beta = 1*np.ones(n_galaxies), shape = n_galaxies)
        sigma_bin = pymc.HalfNormal("sigma_bin", sigma = sigma_bin_prior, shape = n_galaxies)

        # 4) Per-star mixture weight
        w_prior_shared = pymc.Data("w_prior_star", w_prior_star)
        gal_of_star_shared = pymc.Data("gal_of_star", gal_of_star)
        f_star = pymc.Deterministic("f_star", (1.0 - lam_prior) * f_bin[gal_of_star_shared] + lam_prior * w_prior_shared)

        # observation likelihood
        gal_sig_obs = sigma_v[gal_inverse]
        total_single = pymc.math.sqrt(v_errs**2 + gal_sig_obs**2)
        total_binary = pymc.math.sqrt(v_errs**2 + gal_sig_obs**2 + sigma_bin[gal_inverse]**2)

        comp_single = pymc.Normal.dist(mu = v_r[gal_inverse], sigma = total_single)
        comp_binary = pymc.Normal.dist(mu = v_r[gal_inverse], sigma = total_binary)

        f_star_obs = pt.take(f_star, star_inverse)
        w_obs = pymc.math.stack([1.0 - f_star_obs, f_star_obs], axis = 1)

        pymc.Mixture("obs", w = w_obs, comp_dists = [comp_single, comp_binary], observed = vels)

        # sampling
        trace = pymc.sample(draws = draws, tune = tune, chains = chains, target_accept = target_accept, return_inferencedata = True)

    # posterior summaries
    posterior = trace.posterior
    def pct(arr): return np.percentile(arr.values.reshape(-1), [16,50,84])

    mu_vr_vals  = pct(posterior["mu_vr"])
    tau_vr_vals = pct(posterior["tau_vr"])
    mu_sig_vals = pct(posterior["mu_sig"])
    tau_sig_vals= pct(posterior["tau_sig"])

    # galaxy summaries
    v_r_pct = np.zeros((n_galaxies,3))
    s_v_pct = np.zeros((n_galaxies,3))
    f_bin_pct = np.zeros((n_galaxies,3))
    sigma_bin_pct = np.zeros((n_galaxies,3))
    for g in range(n_galaxies):
        v_r_pct[g] = np.percentile(posterior["v_r"][:,:,g].values.ravel(), [16,50,84])
        s_v_pct[g] = np.percentile(posterior["sigma_v"][:,:,g].values.ravel(), [16,50,84])
        f_bin_pct[g] = np.percentile(posterior["f_bin"][:,:,g].values.ravel(), [16,50,84])
        sigma_bin_pct[g] = np.percentile(posterior["sigma_bin"][:,:,g].values.ravel(), [16,50,84])

    # star posterior
    f_star_arr = posterior["f_star"].values
    f_star_pct = np.zeros((n_stars,3))
    for s in range(n_stars):
        f_star_pct[s] = np.percentile(f_star_arr[:,:,s].ravel(), [16,50,84])

    summaries = dict(n_galaxies = n_galaxies, n_stars = n_stars, mu_vr_pct = mu_vr_vals, tau_vr_pct = tau_vr_vals,
                     mu_sig_pct = mu_sig_vals, tau_sig_pct = tau_sig_vals, v_r_percentiles_per_galaxy = v_r_pct,
                     sigma_v_percentiles_per_galaxy = s_v_pct, f_bin_percentiles_per_galaxy = f_bin_pct,
                     sigma_bin_percentiles_per_galaxy = sigma_bin_pct, f_star_percentiles = f_star_pct, 
                     gal_of_star = gal_of_star, star_index_map = unique_stars)

    if return_pvals:
        summaries.update(dict(p_vals_per_star = p_vals, chi2_per_star = chi2_vals,
                              n_epochs_per_star = n_epochs, w_prior_per_star = w_prior_star))

    return model, trace, summaries
