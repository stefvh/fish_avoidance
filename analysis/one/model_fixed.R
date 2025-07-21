data <- read.csv("data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv")


library(car)
library(nortest)
# Example model
library(MASS)
library(plm)

model <- plm(log(speed_fish) ~ distance + speed_robot + cos_apv + cos_rev + min_wall_distance_fish + p_speed_fish, 
            data = data,
            model = "within",
            index = c("experiment_id", "predictor_time"),
            effect = "individual")

print(summary(model))

library(sandwich)
library(lmtest)

# Cluster standard errors at the "state" level
vcov_cluster <- vcovHC(model, type = "HC1", cluster = "group")

# Perform the coeftest with clustered standard errors
coeftest(model, vcov_cluster)

# Extract the fixed effects (individual means)
individual_effects <- fixef(model)

# Calculate the variance of the individual effects
var_individual_effects <- var(individual_effects)

# Calculate the residuals
residuals <- residuals(model)

# Calculate the variance of the residuals
var_residuals <- var(residuals)

# Total variance
total_variance <- var_individual_effects + var_residuals

# Repeatability
repeatability <- var_individual_effects / total_variance
print(repeatability)

n_groups <- length(unique(data$experiment_id))
n_obs <- nrow(data)

# Approximate standard error for the variance components
se_var_individual_effects <- sqrt(2 * var_individual_effects^2 / (n_groups - 1))
se_var_residuals <- sqrt(2 * var_residuals^2 / (n_obs - n_groups))

# Delta method for the repeatability standard error
se_repeatability <- sqrt((se_var_individual_effects / total_variance)^2 + 
                         (var_individual_effects * se_var_residuals / total_variance^2)^2)

# Calculate the 95% confidence interval
ci_lower <- repeatability - 1.96 * se_repeatability
ci_upper <- repeatability + 1.96 * se_repeatability

print(ci_lower)
print(ci_upper)

# # Check for heteroscedasticity
# bptest(model)

# # Check for autocorrelation
# pbgtest(model)


# ## 
# residuals_fe <- residuals(model)

# # # Shapiro-Wilk test
# # shapiro_test <- shapiro.test(residuals_fe)
# # print(shapiro_test)

# # # Anderson-Darling test
# # ad_test <- ad.test(residuals_fe)
# # print(ad_test)

# # # Q-Q Plot
# png("qqplot.png")
# qqPlot(residuals_fe, main = "Q-Q Plot of Residuals")
# dev.off()

# # Histogram of residuals
# png("histogram.png")
# hist(residuals_fe, breaks = 20, main = "Histogram of Residuals", xlab = "Residuals")
# dev.off()


# # Residuals vs Fitted Values
# fitted_values <- as.numeric(fitted(model))
# residuals_fe <- as.numeric(residuals_fe)

# if (is.numeric(fitted_values) && is.numeric(residuals_fe)) {
#   plot(fitted_values, residuals_fe, 
#        xlab = "Fitted values", ylab = "Residuals", 
#        main = "Residuals vs Fitted Values")
#   abline(h = 0, col = "red")
# } else {
#   warning("Fitted values and residuals are not numeric vectors")
# }