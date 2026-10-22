import numpy as np
import scipy.stats as stats

# Function to compute the Fisher Z-transformation and the confidence interval
def pearson_confidence_interval(r, n, confidence_level=0.95):
    # Fisher Z-transformation
    fisher_z = np.arctanh(r)  # Fisher transformation: arctanh(r)
    
    # Compute the standard error
    standard_error = 1 / np.sqrt(n - 3)  # Standard error of the Fisher Z-transformation
    
    # Calculate the Z critical value for the desired confidence level (e.g., 0.95 for 95%)
    z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Calculate the confidence interval in the Fisher Z domain
    z_ci_lower = fisher_z - z_critical * standard_error
    z_ci_upper = fisher_z + z_critical * standard_error
    
    # Convert the confidence interval back to the original correlation scale
    ci_lower = np.tanh(z_ci_lower)  # Inverse Fisher transformation
    ci_upper = np.tanh(z_ci_upper)
    
    return ci_lower, ci_upper

def spearman_confidence_interval(x, y, n_bootstraps=1000):
    bootstrapped_corrs = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(x), len(x))
        x_boot = x[indices]
        y_boot = y[indices]
                
        # Check if the resampled data is constant
        if np.all(x_boot == x_boot[0]) or np.all(y_boot == y_boot[0]):
            continue  # Skip this bootstrap iteration

        boot_corr, _ = stats.spearmanr(x_boot, y_boot)
        bootstrapped_corrs.append(boot_corr)
    
    lower_ci = np.percentile(bootstrapped_corrs, 2.5)
    upper_ci = np.percentile(bootstrapped_corrs, 97.5)
    return lower_ci, upper_ci

def spearman_monte_carlo(x_mean, x_std, y_mean, y_std, n_samples=1000):
    mean_corr = stats.spearmanr(x_mean, y_mean)[0]

    corr_values = []
    for i in range(n_samples):
        # Generate bootstrap samples for each data point
        bootstrap_speed = np.random.normal(x_mean, x_std)
        bootstrap_disp = np.random.normal(y_mean, y_std)
        
        # Compute Spearman correlation for the bootstrap sample
        corr, _ = stats.spearmanr(bootstrap_speed, bootstrap_disp)
        corr_values.append(corr)

    # Compute statistics from bootstrap correlations
    corr_values = np.array(corr_values)
    mean_corr = np.mean(corr_values)
    std_corr = np.std(corr_values)

    # Compute 95% confidence intervals
    ci_lower = np.percentile(corr_values, 2.5)
    ci_upper = np.percentile(corr_values, 97.5)

    # Calculate the two-tailed p-value:
    # The p-value is the proportion of bootstrap correlations that are as extreme as the observed correlation
    extreme_corrs = np.sum(np.abs(corr_values) >= np.abs(mean_corr))
    p_value = extreme_corrs / n_samples

    print("P-value:", p_value)

    return mean_corr, std_corr, ci_lower, ci_upper