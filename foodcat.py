import pandas as pd

# set all columns to be shown
pd.set_option('display.max_columns', None)

df_shak = pd.read_csv("data/foodcat/shak.csv")
# df_shak = df_shak[['_id', 'name']]

df_tim = pd.read_csv("data/foodcat/tim.csv")
# df_tim = df_tim[['_id', 'name']]

# df_foods = pd.merge(df_shak, df_tim, left_on='name', right_on='name')
#
# print(df_foods.head())

# load

# replace shak image by tim image
df_shak['image'] = df_shak['image'].replace(df_shak['image'].tolist(), df_tim['image'].tolist())

# remove __v
df_shak.drop('__v', axis=1, inplace=True)



print(df_shak.head())

# save to csv
df_shak.to_csv("data/foodcat/new.csv", index=False)