# fixed_effects_model.R
library(plm)

# # Load the data from CSV
# df <- read.csv("analysis/0_robot_proximity_fish_behavior/data.csv")
df <- read.csv("analysis/0_robot_proximity_fish_behavior/data.csv")

# Create dummy variables
df$fish_0 <- ifelse(df$experiment_id == 0, 1, 0)
df$fish_1 <- ifelse(df$experiment_id == 1, 1, 0)
df$fish_2 <- ifelse(df$experiment_id == 2, 1, 0)
df$fish_3 <- ifelse(df$experiment_id == 3, 1, 0)
df$fish_4 <- ifelse(df$experiment_id == 4, 1, 0)

print(head(df))

model <- lm(speed_fish ~ distance + speed_robot + fish_0 + fish_1 + fish_2 + fish_3 + time, data = df)

print(summary(model))
print("-------------------")

model <- lm(turn_fish ~ distance + speed_robot + fish_0 + fish_1 + fish_2 + fish_3 + time, data = df)

print(summary(model))
print("-------------------")

model <- lm(min_wall_distance_fish ~ distance + speed_robot + fish_0 + fish_1 + fish_2 + fish_3 + time, data = df)

print(summary(model))

print("-------------------")

model <- lm(high_speed_fish_outlier ~ distance + fish_0 + fish_1 + fish_2 + fish_3, data = df)

print(summary(model))


# Convert dataframe to panel data format
# pdata <- pdata.frame(df, index = c("experiment_id", "frame"))

# Fit the fixed effects model
fixed_effects_model <- plm(speed_fish ~ distance,
                           data = df,
                           index = c("experiment_id", "time"),
                           model = "within")

# Print the summary of the model
print(summary(fixed_effects_model))
