data <- read.csv("data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv")

library(lme4)
library(MuMIn)
control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))

# Example model

data$log_speed_fish <- log(data$speed_fish + 1)

model <- glmer(log_speed_fish ~ scale(distance) + scale(speed_robot) + scale(cos_apv) + scale(cos_rev) + scale(min_wall_distance_fish) + scale(p_speed_fish) + (1 | experiment_id),
    data = data,
    family = inverse.gaussian(link = "log"),
    control = control)

print(summary(model))

library(DHARMa)

# Simulate residuals
simulationOutput <- simulateResiduals(fittedModel = model)

#
# residuals(simulationOutput)

# Plot the simulated residuals
png("residuals_gen_plot.png")
plot(simulationOutput)
dev.off()

# Plot for specific predictor
# plotResiduals(simulationOutput, form = distance)

# Perform additional tests
testUniformity(simulationOutput)
testDispersion(simulationOutput)
testZeroInflation(simulationOutput)

