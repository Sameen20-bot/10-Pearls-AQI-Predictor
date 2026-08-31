from api_functions import connect_hopsworks, day1_model_load, load_blend_model

project = connect_hopsworks()

m1, f1 = day1_model_load(project)
print("Day 1 features:", len(f1))

m2, f2 = load_blend_model(project, 2)
print("Day 2 features:", len(f2))
print("Day 2 parts:", list(m2.keys()))