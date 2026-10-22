
# # Load the data from CSV
df <- read.csv("analysis/0_robot_proximity_fish_behavior/data.csv")
print(head(df))

# Load necessary packages
library(dplyr)
library(data.table)
library(plm)

# Convert to data.table
dt <- as.data.table(df)

# Create a new time variable representing each second
dt[, time_sec := floor(time / 5)]  

# Aggregate data by experiment_id and time_sec
agg_dt <- dt[, .(
  speed_fish = mean(speed_fish),
  distance = mean(distance)
), by = .(experiment_id, time_sec)]

# Create the time trend based on the aggregated time_sec variable
agg_dt[, time_trend := (time_sec - min(time_sec)) / (max(time_sec) - min(time_sec)), by = experiment_id]

# Fit the fixed effects model with the aggregated data
model_plm_trend <- plm(speed_fish ~ log(distance),
                       data = agg_dt,
                       index = c("experiment_id", "time_sec"),
                       model = "within",
                       effect = "twoways")

# Summarize the model
summary(model_plm_trend)

# # Select only necessary columns
# df <- df %>%
#   select(experiment_id, time, speed_fish, distance)

# library(data.table)

# # Convert to data.table
# dt <- as.data.table(df)

# # Perform operations using data.table
# dt[, time_trend := round((time - min(time)) / (max(time) - min(time)), digits=3), by = experiment_id]

# library(plm)

# # Fit the fixed effects model with the continuous time trend
# model_plm_trend <- plm(speed_fish ~ distance, 
#                        data = dt, 
#                        index = c("experiment_id", "time_trend"), 
#                        model = "within",
#                        effect = "twoways")

# # Summarize the model
# summary(model_plm_trend)


# library(lme4)
# # Random intercepts and random slopes for time within experiment_id
# model <- lmer(speed_fish ~ distance + (1 + time | experiment_id), data = df)

# print(summary(model))

### USING PLM
# library(plm)

# m_plm <- plm(speed_fish ~ distance + factor(time),
#     data = df,
#     index = c("experiment_id", "time"),
#     model = "within",
# )

# print(summary(m_plm))

# m_plm <- plm(speed_fish ~ distance,
#     data = df,
#     index = c("experiment_id"),
#     model = "within"
# )

# print(summary(m_plm))

# library(feisr)
# m_feis <- feis(speed_fish ~ distance | time,
#     data = df, 
#     id = "experiment_id")

# print(summary(m_feis))

# library(feisr)
# m_feis <- feis(speed_fish ~ distance | time,
#     data = df, 
#     id = c("experiment_id", "time")
# )

# print(summary(m_feis))