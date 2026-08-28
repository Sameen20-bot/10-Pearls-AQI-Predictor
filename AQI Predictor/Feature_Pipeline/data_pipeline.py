from functions import (weather_fetch, aqi_fetch, merge_and_clean,
                       building_hourly_features, building_day1_features,
                       building_day2_day3_features, connect_hopsworks,
                       push_to_hopsworks)

print("Step1: Fetching the Data:")
fetching_weather = weather_fetch()
fetching_aqi = aqi_fetch()
print("Step1 is Done :)")


print("Step2: Merging and Cleaning:")
Merging_Cleaning = merge_and_clean(fetching_weather, fetching_aqi)
print("Step2 is Done :)")


print("Step3: Building hourly features:")
Hourly_Features = building_hourly_features(Merging_Cleaning)
print("Step3 is Done :)")


print("Step4: Building Daily features for Day 1:")
Day1_Features = building_day1_features(Hourly_Features)
print("Step4 is Done :)")


print("Step5: Building Daily features for Day 2 and Day 3:")
Day2_Features, Day3_Features = building_day2_day3_features(Hourly_Features)
print("Step5 is Done :)")


print("Step6: Connecting and Pushing to Hopsworks:")
Connecting_Hopsworks = connect_hopsworks()
Pushing_Hopsworks = push_to_hopsworks(Connecting_Hopsworks, Day1_Features, Day2_Features, Day3_Features)
print("Step6 is Done :)")

print("Data Pipeline Completed Yuppy ;)")