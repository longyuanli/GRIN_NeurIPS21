import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist
from mmd import CMMD_3_Sample_Test
'''
Functions for calculating metrics given predicted data and ground truth
'''

def calc_metrics(results):
    # shape [num_batches, num_prediction_steps, num_vars, 2]
    loc_target = results['loc_targets']
    # shape [num_batches, train_steps, num_vars, 2]
    loc_past = results['loc_pasts']
    kl = results['kl_avg']
    sigma_x = results['sigma_x']
    # shape [batch_size, prediction_steps,  num_vars, num_sample, input_size]
    q_loc_pred = results['q_loc_pred']
    p_loc_pred = results['p_loc_pred']
    num_data = p_loc_pred.shape[0]
    num_sample = p_loc_pred.shape[3]
    loc_target = np.stack([loc_target]*num_sample, axis=3)

    # for z ~ p(z|x_P) prediction error
    mse_error = (p_loc_pred - loc_target)**2
    # shape [batch_size, num_vars, prediction_steps, num_sample]
    euclidean_rmse = np.sqrt(mse_error.sum(-1))
    mse_avg = mse_error.mean((1, 2, 4)).min(1).mean()
    FDE = euclidean_rmse[:, -1, :, :].mean(1).min(1).mean()
    ADE = euclidean_rmse.mean((1, 2)).min(1).mean()

    # for z ~ q(z|x_P, x_F) prediction error
    q_mse_error = (q_loc_pred - loc_target)**2
    # shape [batch_size, num_vars, prediction_steps, num_sample]
    q_euclidean_rmse = np.sqrt(q_mse_error.sum(-1))
    q_mse_avg = q_mse_error.mean((1, 2, 4)).min(1).mean()
    q_FDE = q_euclidean_rmse[:, -1, :, :].mean(1).min(1).mean()
    q_ADE = q_euclidean_rmse.mean((1, 2)).min(1).mean()

    # Compute ELBO = NLL + KL
    nll = - norm.logpdf(loc_target, loc=q_loc_pred, scale=sigma_x)
    ELBO = nll.sum((1, 2, 4)).mean() + kl

    # Compute conditional MMD using the relative test script.
    # For a model comparison test, replace the indicated argument with predictions from a different model.
    # Without any changes, this test should never reject (p>0.05) because we compare the model prediction with the ground truth.
    # In order to make a consistent distance comparison for each test dataset,
    # the bandwidths are chosen using only the median distances in the test dataset.
    sigma1 = np.median(pdist(loc_past.reshape(num_data, -1)))
    sigma2 = np.median(pdist(loc_target.reshape(num_data, -1)))
    pvalue, tstat, MMDXY, MMDXZ = CMMD_3_Sample_Test(loc_past.reshape(num_data, -1),
                                                     loc_target.reshape(
                                                         num_data, -1),
                                                     # can replace this with predictions from another model for model comparison test.
                                                     loc_target.reshape(
                                                         num_data, -1),
                                                     # if comparing another model with GRIN, then set this to the predictions of GRIN
                                                     p_loc_pred.reshape(
                                                         num_data, -1),
                                                     sigma1=sigma1, sigma2=sigma2)
    return {
        "MSE": mse_avg,
        "FDE": FDE,
        "ADE": ADE,
        "q_MSE": q_mse_avg,
        "q_FDE": q_FDE,
        "q_ADE": q_ADE,
        "ELBO": ELBO,
        "KL": kl,
        "pvalue": pvalue,
        "tstat": tstat,
        "MMDXY": MMDXY,
        "MMDXZ": MMDXZ
    }


def calc_metrics_debug(results):
    print("Calulating Metics")
    # shape [num_batches, num_prediction_steps, num_vars, 2]
    loc_target = results['loc_targets']
    # shape [num_batches, train_steps, num_vars, 2]
    loc_past = results['loc_pasts']
    kl = results['kl_avg']
    sigma_x = results['sigma_x']
    # shape [batch_size, prediction_steps,  num_vars, num_sample, input_size]
    q_loc_pred = results['q_loc_pred']
    p_loc_pred = results['p_loc_pred']
    num_data = p_loc_pred.shape[0]
    num_sample = p_loc_pred.shape[3]
    #loc_target = np.stack([loc_target]*num_sample, axis=3)
    loc_target = np.expand_dims(loc_target,axis=3)

    # for z ~ p(z|x_P) prediction error
    mse_error = (p_loc_pred - loc_target)**2
    # shape [batch_size, num_vars, prediction_steps, num_sample]
    euclidean_rmse = np.sqrt(mse_error.sum(-1))
    mse_avg = mse_error.mean((1, 2, 4)).min(1).mean()
    FDE = euclidean_rmse[:, -1, :, :].mean(1).min(1).mean()
    ADE = euclidean_rmse.mean((1, 2)).min(1).mean()
    #FDE = euclidean_rmse[:, -1, :, :].min(-1).mean()
    #ADE = euclidean_rmse.mean(2).min(-1).mean()
    # for z ~ q(z|x_P, x_F) prediction error
    q_mse_error = (q_loc_pred - loc_target)**2
    # shape [batch_size, num_vars, prediction_steps, num_sample]
    q_euclidean_rmse = np.sqrt(q_mse_error.sum(-1))
    q_mse_avg = q_mse_error.mean((1, 2, 4)).min(1).mean()
    q_FDE = q_euclidean_rmse[:, -1, :, :].mean(1).min(1).mean()
    q_ADE = q_euclidean_rmse.mean((1, 2)).min(1).mean()

    # Compute ELBO = NLL + KL
    nll = - norm.logpdf(loc_target, loc=q_loc_pred, scale=sigma_x)
    ELBO = nll.sum((1, 2, 4)).mean() + kl
    return {
        "MSE": mse_avg,
        "FDE": FDE,
        "ADE": ADE,
        "q_MSE": q_mse_avg,
        "q_FDE": q_FDE,
        "q_ADE": q_ADE,
        "ELBO": ELBO,
        "KL": kl
    }