import pandas as pd

# set all columns to be shown
pd.set_option('display.max_columns', None)

# from foods get _id and name columns
df_foods = pd.read_csv("data/foods.csv")
df_foods = df_foods[['_id', 'name']]

# from foodDetails get _id, food_name
df_foodDetails = pd.read_csv("data/foodDetails.csv")
df_foodDetails = df_foodDetails[['_id', 'food_name']]

# merge df_foods and df_foodDetails
df_foods = pd.merge(df_foods, df_foodDetails, left_on='name', right_on='food_name')

print(df_foods.head())

# load orderItemWithQuanties
df_orderItemWithQuantities = pd.read_csv("data/orderItemWithQuantities.csv")

# replace food with the _id_x from df_foods by _id_y
df_orderItemWithQuantities['food'] = df_orderItemWithQuantities['food'].replace(df_foods['_id_y'].tolist(), df_foods['_id_x'].tolist())

print(df_orderItemWithQuantities.head())

# save to csv
df_orderItemWithQuantities.to_csv("data/orderItemWithQuantities-gen.csv", index=False)