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

# ### USING LM

# model <- lm(speed_fish ~ distance + fish_0 + fish_1 + fish_2 + fish_3 + time, data = df)

# print(summary(model))

# print("-------------------")

# ### USING PLM
# library(plm)

# # distance
# model <- plm(speed_fish ~ distance,
#     data = df,
#     index = c("experiment_id", "time"),
#     model = "within"
# )

# print(summary(model))

# print("-------------------")

# # log(distance)
# model <- plm(speed_fish ~ log(distance),
#     data = df,
#     index = c("experiment_id", "time"),
#     model = "within"
# )

# print(summary(model))

### USING GLMM
library(lme4)
library(MuMIn)

# Rescale the distance variable
df$scaled_distance <- scale(df$distance)

# scale_info <- attr(df$scaled_distance, "scaled:scale")
# print(scale_info)  # This is the standard deviation of the original distance variable

# Increase the number of iterations and use different optimizers
control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# # Gamma
# model <- glmer(speed_fish ~ scaled_distance + (1 | experiment_id),
#     data = df,
#     family = Gamma(),
#     control = control
# )

# print(summary(model))

# r2 <- r.squaredGLMM(model)
# print(r2)

# print("-------------------")


# # Gaussian
# model <- glmer(speed_fish ~ scaled_distance + (1 | experiment_id),
#     data = df,
#     family = gaussian(link = "log"),
#     control = control
# )

# print(summary(model))

# r2 <- r.squaredGLMM(model)
# print(r2)

# print("-------------------")

# # Inverse Gaussian
# model <- glmer(speed_fish ~ scaled_distance + (1 | experiment_id) + (1 | time),
#     data = df,
#     family = inverse.gaussian(link = "log"),
#     control = control
# )

# print(summary(model))

# r2 <- r.squaredGLMM(model)
# print(r2)

print("-------------------")

# Inverse Gaussian
model <- glmer(speed_fish ~ scaled_distance + scale(time) + (1 + time || experiment_id),
    data = df,
    family = inverse.gaussian(link = "log"),
    control = control
)

print(summary(model))

r2 <- r.squaredGLMM(model)
print(r2)

# print("-------------------")

# model_null <- glmer(speed_fish ~ 1 + (1 | experiment_id),
#                     data = df,
#                     family = inverse.gaussian(link = "log"),
#                     control = control)


# # Inverse Gaussian
# model_full <- glmer(speed_fish ~ scaled_distance + (1 | experiment_id),
#     data = df,
#     family = inverse.gaussian(link = "log"),
#     control = control
# )

# anova(model_null, model_full)

# AIC(model_null, model_full)
# BIC(model_null, model_full)

# print(summary(model_full))

# r2 <- r.squaredGLMM(model_full)
# print(r2)

# print("-------------------")

