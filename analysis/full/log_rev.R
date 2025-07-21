library(lme4)

data <- read.csv("test_log_rev_data.csv")

# b_cos_rev = 1 if cos_rev > 0 else 0
data$b_cos_rev <- ifelse(data$rv_cos_rev > 0, 1, 0)
# data$apv <- data$cos_apv * data$speed_robot

# data$experiment_id <- as.factor(data$experiment_id)
# data$predictor_time <- as.factor(data$predictor_time)

anyNA(data)

start_vals <- c(0, 0, 0)  # Adjust according to the number of predictors

model <- glmer(b_cos_rev ~ distance + apv + (1 | experiment_id), 
            data = data,
            family = binomial(link = "logit"))

print(summary(model))

# Step 1: Create a grid of values for distance and speed_robot
threshold_d_values <- seq(min(data$distance), max(data$distance), length.out = 100)
threshold_s_values <- seq(min(data$apv), max(data$apv), length.out = 100)

# Create a data frame with all combinations of distance and speed_robot
prediction_grid <- expand.grid(distance = threshold_d_values, apv = threshold_s_values)

# Step 2: Predict probabilities
predicted_probs <- predict(model, newdata = prediction_grid, type = "response", re.form = NA)

# Add predicted probabilities to the grid
prediction_grid$predicted_prob <- predicted_probs

# Step 3: Find thresholds where predicted_prob > 0.95 (or another threshold)
threshold_95 <- subset(prediction_grid, predicted_prob > 0.95)
threshold_90 <- subset(prediction_grid, predicted_prob > 0.90)

# View the results
head(threshold_95)
head(threshold_90)

# If you want to visualize it
library(ggplot2)
ggplot(prediction_grid, aes(x = distance, y = apv, fill = predicted_prob)) +
  geom_tile() +
  geom_contour(aes(z = predicted_prob), breaks = c(0.90, 0.95), color = "red") +
  scale_fill_gradient2(low = "blue", high = "red", midpoint = 0.5, limits = c(0, 1)) +
  labs(title = "Predicted Probability of b_cos_rev = 1", x = "Distance", y = "Robot's Approach Strength") +
  theme_minimal()