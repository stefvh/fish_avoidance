data <- read.csv("data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv")

library(lmerTest)
# Example model

model <- lmer(speed_fish ~ distance + speed_robot + cos_apv + cos_rev + min_wall_distance_fish + p_speed_fish + (1 | experiment_id), data = data)

print(summary(model))

library(DHARMa)

# Simulate residuals
simulationOutput <- simulateResiduals(fittedModel = model)

# residuals(simulationOutput)

# QQ plot residuals
library(car)
png("qqplot_gen.png")
residuals <- residuals(model)
qqPlot(residuals, main = "Q-Q Plot of Residuals")
dev.off()

# Plot the simulated residuals
png("residuals_plot.png")
plot(simulationOutput)
dev.off()

# Plot for specific predictor
plotResiduals(simulationOutput)

# Perform additional tests
testUniformity(simulationOutput)
testDispersion(simulationOutput)
testZeroInflation(simulationOutput)

#
# library(rptR)

# # Fit the mixed-effects model using rptR
# rpt_result <- rpt(speed_fish ~ distance + speed_robot + cos_apv + cos_rev + 
#                   min_wall_distance_fish + p_speed_fish + 
#                   (1 | experiment_id), 
#                   grname = "experiment_id", 
#                   data = data, datatype = "Gaussian", nboot = 1000, npermut = 1000)

# # Print the repeatability results
# print(rpt_result)