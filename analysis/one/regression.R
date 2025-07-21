data <- read.csv("data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv")

library(plm)

model <- plm(speed_fish ~ distance + speed_robot + cos_apv + min_wall_distance_fish,
    data = data,
    index = c("experiment_id"),
    model = "within",
    effect = "individual"
)

print(summary(model))

print("-----------------")

model <- plm(acceleration_fish ~ distance + speed_robot + cos_apv + min_wall_distance_fish,
    data = data,
    index = c("experiment_id"),
    model = "within",
    effect = "individual"
)

print(summary(model))