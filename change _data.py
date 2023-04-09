import pandas as pd

# show all columns
pd.set_option('display.max_columns', None)

data_path = "data/"

# load order_data
order_data = pd.read_csv('order_data_2.csv')

# load preprocessed data
resturant_data = pd.read_csv(data_path + 'pre_processed.csv')


# print column names
print(order_data.columns)
print()
print(resturant_data.columns)

# replace order data with resturant data
order_data['user_id'] = resturant_data['orderedBy']
order_data['age'] = resturant_data['age']
order_data['cuisine'] = resturant_data['food_cuisine']
order_data['food_id'] = resturant_data['food']
order_data['food_name'] = resturant_data['food_name']
order_data['food_type'] = resturant_data['food_type']
order_data['Ingredients'] = resturant_data['ingredients']
order_data['food_rating'] = resturant_data['feedback']

# rename resturant data column
resturant_data.rename(columns={
    'orderedBy': 'user_id',
    'age': 'age',
    'food_cuisine': 'cuisine',
    'food': 'food_id',
    'food_name': 'food_name',
    'food_type': 'food_type',
    'ingredients': 'Ingredients',
    'feedback': 'food_rating'
}, inplace=True)



# save order data
order_data.to_csv('order_data_3.csv', index=False)

# save resturant data
resturant_data.to_csv('resturant_data_2.csv', index=False)
