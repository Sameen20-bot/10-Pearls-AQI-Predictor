from train_functions import (connect_hopsworks, read_feature_group,
                             prepare_data_day1, prepare_data_day2_3,
                             day1_train, day2_day3_train,
                             predict_blend, metrics_get,
                             baseline_R2_Score,
                             register_day1_model, register_day2_day3_model)

import time


print("Step1: Connecting to Hopsworks")
project = connect_hopsworks()
print("Step1: Done Yuppy ;)")

print("------------------DAY ONE--------------------")

print("Step2: Reading data from feature group")
df = read_feature_group(project, "aqi_daily_day1")
print("Step2: Done Yuppy ;)")

print("Step3: Preparing the data for day One")
X_train1, X_test1, Y_train1, Y_test1 = prepare_data_day1(df)
print("Step3: Done Yuppy ;)")

print("Step4: Training the Day 1 XGB Boost Model")
model1 = day1_train(X_train1, Y_train1)
pred1 = model1.predict(X_test1)
print("Step4: Done Yuppy ;)")

print("Step5: Now get all the metrics")
metrics = metrics_get(Y_test1, pred1)
print("Step5: Done Yuppy ;)")

print("Step6: Now check Baseline Score")
baseline = baseline_R2_Score(X_test1, Y_test1)
print("Step6: Done Yuppy ;)")

print(f"R2: {metrics['r2']} \n Baseline: {baseline}")

print("Step7: Saving Day1 Model")
register_day1_model(project, model1, metrics,  X_train1.columns)
print("Step7: Done Yuppy ;)")

time.sleep(15)  


print("------------------DAY TWO--------------------")

print("Step2: Reading data from feature group")
df2 = read_feature_group(project, "aqi_daily_day2")
print("Step2: Done Yuppy ;)")

print("Step3: Preparing the data for day Two")
X_train2, X_test2, Y_train2, Y_test2 = prepare_data_day2_3(df2, 2)
print("Step3: Done Yuppy ;)")

print("Step4: Training the Day 2 from 5 Blend Models")
model2 = day2_day3_train(X_train2, Y_train2)
pred2 = predict_blend(model2, X_test2)
print("Step4: Done Yuppy ;)")

print("Step5: Now get all the metrics")
metrics2 = metrics_get(Y_test2, pred2)
print("Step5: Done Yuppy ;)")

print("Step6: Now check Baseline Score")
baseline2 = baseline_R2_Score(X_test2, Y_test2)
print("Step6: Done Yuppy ;)")

print(f"R2: {metrics2['r2']} \n Baseline: {baseline2}")

print("Step7: Saving Day2 Model")
register_day2_day3_model(project, model2, metrics2,  X_train2.columns, 2)
print("Step7: Done Yuppy ;)")

time.sleep(15)  

print("------------------DAY THREE--------------------")

print("Step2: Reading data from feature group")
df3 = read_feature_group(project, "aqi_daily_day3")
print("Step2: Done Yuppy ;)")

print("Step3: Preparing the data for day Three")
X_train3, X_test3, Y_train3, Y_test3 = prepare_data_day2_3(df3, 3)
print("Step3: Done Yuppy ;)")

print("Step4: Training the Day 3 from 5 Blend Models")
model3 = day2_day3_train(X_train3, Y_train3)
pred3 = predict_blend(model3, X_test3)
print("Step4: Done Yuppy ;)")

print("Step5: Now get all the metrics")
metrics3 = metrics_get(Y_test3, pred3)
print("Step5: Done Yuppy ;)")

print("Step6: Now check Baseline Score")
baseline3 = baseline_R2_Score(X_test3, Y_test3)
print("Step6: Done Yuppy ;)")

print(f"R2: {metrics3['r2']} \n Baseline: {baseline3}")

print("Step7: Saving Day3 Model")
register_day2_day3_model(project, model3, metrics3,  X_train3.columns, 3)
print("Step7: Done Yuppy ;)")

print("\nTraining Pipeline Completed")

