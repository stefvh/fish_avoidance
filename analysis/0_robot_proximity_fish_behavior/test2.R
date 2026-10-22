library(dplyr)
library(data.table)

# # Load the data from CSV
df <- read.csv("analysis/0_robot_proximity_fish_behavior/data.csv")

# Create a new time variable representing each second before lagging
df <- df %>%
  mutate(time_sec = floor(time / 5))

# Create lagged variables for different time lags before aggregating
df <- df %>%
  group_by(experiment_id) %>%
  mutate(distance_lag1 = lag(distance, 5),
         distance_lag2 = lag(distance, 10),
         distance_lag3 = lag(distance, 15))

# Convert to data.table
dt <- as.data.table(df)

# Aggregate data by experiment_id and time_sec
agg_dt <- dt[, .(
  speed_fish = mean(speed_fish, na.rm = TRUE),
  distance = mean(distance, na.rm = TRUE),
  distance_lag1 = mean(distance_lag1, na.rm = TRUE),
  distance_lag2 = mean(distance_lag2, na.rm = TRUE),
  distance_lag3 = mean(distance_lag3, na.rm = TRUE)
), by = .(experiment_id, time_sec)]

# Ensure lagged variables were correctly created
head(agg_dt)

library(plm)
library(car)

# Check pairwise correlations
cor_matrix <- cor(agg_dt[, c("distance", "distance_lag1", "distance_lag2", "distance_lag3")], use = "complete.obs")
print(cor_matrix)

# Check for multicollinearity
vif_data <- lm(speed_fish ~ distance + distance_lag1 + distance_lag2 + distance_lag3, data = agg_dt)
vif_out <- vif(vif_data)
print(vif_out)

# Fit models with different lags to find the best fit
model_lag0 <- plm(speed_fish ~ distance,
                  data = agg_dt,
                  model = "within",
                  index = c("experiment_id", "time_sec"),
                  effect = "individual")
model_lag1 <- plm(speed_fish ~ distance + distance_lag1,
                  data = agg_dt,
                  model = "within",
                  index = c("experiment_id", "time_sec"),
                  effect = "individual")
model_lag2 <- plm(speed_fish ~ distance + distance_lag1 + distance_lag2,
                  data = agg_dt,
                  model = "within",
                  index = c("experiment_id", "time_sec"),
                  effect = "individual")
model_lag3 <- plm(speed_fish ~ distance + distance_lag1 + distance_lag2 + distance_lag3,
                  data = agg_dt,
                  model = "within",
                  index = c("experiment_id", "time_sec"),
                  effect = "individual")

# Extract summaries
summary_lag0 <- summary(model_lag0)
summary_lag1 <- summary(model_lag1)
summary_lag2 <- summary(model_lag2)
summary_lag3 <- summary(model_lag3)

print(summary_lag1)
print(summary_lag2)
print(summary_lag3)

# Manually create a comparison table
create_comparison_table <- function(summary_obj, model_name) {
  coefs <- summary_obj$coefficients
  data.frame(
    Model = model_name,
    Term = rownames(coefs),
    Estimate = coefs[, "Estimate"],
    Std.Error = coefs[, "Std. Error"],
    t.value = coefs[, "t-value"],
    p.value = coefs[, "Pr(>|t|)"]
  )
}

results_lag0 <- create_comparison_table(summary_lag0, "Lag 0")
results_lag1 <- create_comparison_table(summary_lag1, "Lag 1")
results_lag2 <- create_comparison_table(summary_lag2, "Lag 2")
results_lag3 <- create_comparison_table(summary_lag3, "Lag 3")

results_comparison <- bind_rows(results_lag0, results_lag1, results_lag2, results_lag3)

# Print the comparison table
print(results_comparison)

# Extract Adjusted R-squared
adj_r_squared <- data.frame(
  Model = c("Lag 0 ", "Lag 1", "Lag 2", "Lag 3"),
  Adjusted_R2 = c(summary_lag0$r.squared[2], summary_lag1$r.squared[2], summary_lag2$r.squared[2], summary_lag3$r.squared[2])
)

print(adj_r_squared)