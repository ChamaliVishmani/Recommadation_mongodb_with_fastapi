import os
from datetime import datetime

import pandas as pd

from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import motor.motor_asyncio

import requests

# show all columns
pd.set_option('display.max_columns', None)

app = FastAPI()

# load environment variables
from dotenv import load_dotenv

load_dotenv()

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.food_test

data_length = 100000


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# const foodDetailsSchema = new Schema(
#     food_id: { type: Schema.Types.ObjectId, ref: "User" },
#     food_name: { type: String },
#     food_type: { type: String },
#     Ingredients: { type: String },
#   { collection: "foodDetails" }

class FoodDetailsModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    food_name: str = Field(...)
    food_type: str = Field(...)
    food_cuisine: str = Field(...)
    ingredients: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "food_name": "Fish and Chips",
                "food_type": "Main",
                "food_cuisine": "British",
                "ingredients": "Fish, Chips, Salt, Vinegar",
            }
        }


# List all foods
@app.get("/fooddetails", response_description="List all foods", response_model=List[FoodDetailsModel])
async def list_foods():
    foods = await db["foodDetails"].find().to_list(data_length)
    return foods


# const orderItemsSchema = new Schema(
#     orderID: { type: Schema.Types.ObjectId, ref: "Order" },
#     food: { type: Schema.Types.ObjectId, ref: "Food" },
#     quantity: { type: Number },
#     price: { type: Number },
#   { collection: "orderItemWithQuantities" }


class OrderItemWithQuantityModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    orderID: PyObjectId = Field(default_factory=PyObjectId)
    food: PyObjectId = Field(default_factory=PyObjectId)
    quantity: int
    price: int

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "orderID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "food": "60f4d5c5b5f0f0e5e8b2b5c9",
                "quanitity": 2,
                "price": 20
            }
        }


@app.get(
    "/orderItemWithQuantities", response_description="List all order items with quantities",
    response_model=List[OrderItemWithQuantityModel]
)
async def list_order_items_with_quantities():
    order_items_with_quantities = await db["orderItemWithQuantities"].find().to_list(data_length)
    return order_items_with_quantities


# const orderSchema = new Schema(
#     createDate: { type: String },
#     createTime: { type: String },
#     status: { type: String },
#     orderedBy: { type: Schema.Types.ObjectId, ref: "User" },
#     billValue: { type: String },
#     discount: { type: String },
#     orderType: { type: Schema.Types.ObjectId, ref: "OrderType" },
#     table: { type: Schema.Types.ObjectId, ref: "Table" },
#     handleBy: { type: Schema.Types.ObjectId, ref: "Employee" },
#   { collection: "orders" }


class OrderModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createDate: str = Field(...)
    createTime: str = Field(...)
    status: str = Field(...)
    orderedBy: PyObjectId = Field(default_factory=PyObjectId)
    billValue: str = Field(...)
    discount: str = Field(...)
    orderType: str = Field(...)
    table: str = Field(...)
    handleBy: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "createDate": "2021-07-21",
                "createTime": "18:11:10",
                "status": "pending",
                "orderedBy": "60f4d5c5b5f0f0e5e8b2b5c9",
                "billValue": "100",
                "discount": "10",
                "orderType": "60f4d5c5b5f0f0e5e8b2b5c9",
                "table": "60f4d5c5b5f0f0e5e8b2b5c9",
                "handleBy": "60f4d5c5b5f0f0e5e8b2b5c9"
            }
        }


@app.get("/orders", response_description="List all orders", response_model=List[OrderModel])
async def list_orders():
    orders = await db["orders"].find().to_list(data_length)
    return orders


# const userSchema = new Schema(
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     firstName: { type: String },
#     lastName: { type: String },
#     userName: { type: String },
#     email: { type: String },
#     dateOfBirth: { type: String },
#     mobileNumber: { type: String },
#     password: { type: String },
#   { collection: "users" }

class UserModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    firstName: str
    lastName: str
    userName: str
    email: str
    dateOfBirth: str
    mobileNumber: str
    password: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "userID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "firstName": "John",
                "lastName": "Doe",
                "userName": "JohnDoe",
                "email": "johndoe@example.com",
                "dateOfBirth": "1990-01-01",
                "mobileNumber": "0712345678",
                "password": "password"
            }
        }


@app.get("/users", response_description="List all users", response_model=List[UserModel])
async def list_users():
    users = await db["users"].find().to_list(data_length)
    return users


# const feedbackSchema = new Schema(
#     feedback: { type: String },
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     orderId: { type: Schema.Types.ObjectId, ref: "Order" },
#   { collection: "feedbacks" }

class FeedbackModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    orderID: PyObjectId = Field(default_factory=PyObjectId)
    feedback: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "orderID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "feedback": 2
            }
        }


@app.get("/feedbacks", response_description="List all feedbacks", response_model=List[FeedbackModel])
async def list_feedbacks():
    feedbacks = await db["feedbacks"].find().to_list(data_length)
    return feedbacks


# 1. Aggregate orders with orderItemWithQuantities
# 2. for each orderItem get orderedBy and for that get dateOfBirth from users
# 3. for each orderItem get food_id, food_type, Ingredients from foodDetails by food_name
#
# user_id        :  orders.orderedBy
# age            :  users.dateOfBirth
# food_id        :  orderItemWithQuantities._id
# food_name      :  foodDetails.food
# food_type      :  foodDetails.food_type
# Ingredients    :  foodDetails.Ingredients
# food_rating    :  feedbacks.feedback

# FoodDetailsModel
# OrderItemWithQuantityModel
# OrderModel
# UserModel
# FeedbackModel

# take data from orders, orderItemWithQuantities, foodDetails, users, feedbacks and aggregate them
# 1. Aggregate orders with orderItemWithQuantities
# 2. for each orderItem get orderedBy and for that get dateOfBirth from users
# 3. for each orderItem get food_id, food_type, Ingredients from foodDetails by food_name


#   {
#     "_id": "64315e59362c27c707fe156b",
#     "food_name": "Pol Roti with chili salad ",
#     "food_type": "Vegetarian",
#     "food_cuisine": "Sri Lankan",
#     "ingredients": "Flour\nCoconut\nOnion\nGreen chili\nSri Lankan spices"
#   },


#   {
#     "_id": "64316a3f362c27c707fe18a9",
#     "orderID": "64316105362c27c707fe15ec",
#     "food": "64315e59362c27c707fe1586",
#     "quantity": 2,
#     "price": 885
#   },


# {
#     "_id": "64316ada362c27c707fe20c6",
#     "createDate": "1970-10-20",
#     "createTime": "18:11:10",
#     "status": "pending",
#     "orderedBy": "64315d86362c27c707fe155c",
#     "billValue": "774",
#     "discount": "94",
#     "orderType": "1",
#     "table": "2",
#     "handleBy": "8"
#   },


# {
#     "_id": "64315d86362c27c707fe1529",
#     "firstName": "Justin",
#     "lastName": "Diaz",
#     "userName": "JustinDiaz",
#     "email": "Justin.Diaz@gmail.com",
#     "dateOfBirth": "2005-06-29",
#     "mobileNumber": "(326)149-8817",
#     "password": "v^5BXk2C"
#   },


#   {
#     "_id": "64316b59362c27c707fe2386",
#     "feedback": "2",
#     "orderID": null
#   },


base_url = "http://localhost:8000/"
save_path = "data/"


def call_api():
    url_list = ["orderItemWithQuantities", "orders", "fooddetails", "users", "feedbacks"]

    for url in url_list:
        url = base_url + url
        response = requests.get(url).json()
        # convert to dataframe
        # If using all scalar values, you must pass an index
        df = pd.DataFrame(response)
        # save to csv
        df.to_csv(save_path + url.split("/")[-1] + ".csv", index=False)

        print("\n\nName of the file: ", url.split("/")[-1])
        print(df.head())


def aggregate_data():
    # read csv
    orderItemWithQuantities_df = pd.read_csv(save_path + "orderItemWithQuantities.csv")
    orders_df = pd.read_csv(save_path + "orders.csv")
    foodDetails_df = pd.read_csv(save_path + "fooddetails.csv")
    users_df = pd.read_csv(save_path + "users.csv")
    feedbacks_df = pd.read_csv(save_path + "feedbacks.csv")

    # rename _id to id in all dataframes
    # orderItemWithQuantities_df.rename(columns={'_id': 'id'}, inplace=True)
    # orders_df.rename(columns={'_id': 'id'}, inplace=True)
    # foodDetails_df.rename(columns={'_id': 'id'}, inplace=True)
    # users_df.rename(columns={'_id': 'id'}, inplace=True)
    # feedbacks_df.rename(columns={'_id': 'id'}, inplace=True)

    # keep only required columns
    # ['_id', 'orderID', 'food']
    orderItemWithQuantities_df = orderItemWithQuantities_df[['_id', 'orderID', 'food']]
    # ['_id', 'orderedBy', 'orderType']
    orders_df = orders_df[['_id', 'orderedBy', 'orderType']]
    # ['_id', 'food_name', 'food_type', 'food_cuisine', 'ingredients']
    foodDetails_df = foodDetails_df[['_id', 'food_name', 'food_type', 'food_cuisine', 'ingredients']]
    # ['_id', 'dateOfBirth']
    users_df = users_df[['_id', 'dateOfBirth']]
    # ['_id', 'orderID', 'feedback']
    feedbacks_df = feedbacks_df[['_id', 'orderID', 'feedback']]

    # rename orderWithQuantities_df _id to orderItemID
    orderItemWithQuantities_df.rename(columns={'_id': 'orderItemID'}, inplace=True)

    # print list of columns in each dataframe
    print("orderItemWithQuantities_df: ", orderItemWithQuantities_df.columns)
    print("orders_df: ", orders_df.columns)
    print("foodDetails_df: ", foodDetails_df.columns)
    print("users_df: ", users_df.columns)
    print("feedbacks_df: ", feedbacks_df.columns)

    # add orders_df to orderWithQuantities_df by id to orderID
    df = pd.merge(orderItemWithQuantities_df, orders_df, left_on='orderID', right_on='_id')

    if '_id' in df.columns:
        print(df[df['orderID'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add foodDetails_df to df by id to food
    df = pd.merge(df, foodDetails_df, left_on='food', right_on='_id')
    if '_id' in df.columns:
        print(df[df['food'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add users_df to df by id to orderedBy
    df = pd.merge(df, users_df, left_on='orderedBy', right_on='_id')
    if '_id' in df.columns:
        print(df[df['orderedBy'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add feedbacks_df to df by id to orderID
    df = pd.merge(df, feedbacks_df, left_on='orderID', right_on='orderID')
    if '_id' in df.columns:
        print(df[df['orderID'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # save to csv
    df.to_csv(save_path + "aggregate.csv", index=False)

    print("\n\nName of the file: aggregate")
    print(df.head())


def process_data():
    # load csv
    df = pd.read_csv(save_path + "aggregate.csv")

    # convert dateOfBirth to age
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'])
    print(df['dateOfBirth'].head())
    # now date in 2002-08-30 format
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d")

    df['age'] = pd.to_datetime(now) - df['dateOfBirth']
    df.drop(columns=['dateOfBirth'], inplace=True)

    # convert age to years (int)
    df['age'] = df['age'].dt.days / 365
    df['age'] = df['age'].astype(int)

    # save to csv
    df.to_csv(save_path + "processed.csv", index=False)

def analyze_data():
    # load csv
    df = pd.read_csv(save_path + "processed.csv")

    # data types
    print(df.dtypes)

    # null values
    print(df.isnull().sum())

    # duplicate values
    print(df.duplicated().sum())

    # unique values
    print(df.nunique())

    # describe
    print(df.describe())


def pre_process():
    # load csv
    df = pd.read_csv(save_path + "processed.csv")

    # remove column if 75% of the values are null
    df.dropna(thresh=len(df) * 0.25, axis=1, inplace=True)

    # remove null values
    df.dropna(inplace=True)

    # remove null values
    df.dropna(inplace=True)

    # remove duplicate values
    df.drop_duplicates(inplace=True)


    # save to csv
    df.to_csv(save_path + "pre_processed.csv", index=False)


# def

from recommand import get_rec

# my_profile = "64315d86362c27c707fe155c"
my_profile = "64315d86362c27c707fe152z"
#
if __name__ == "__main__":
    # call_api()
    # aggregate_data()
    # process_data()
    # analyze_data()
    # pre_process()
    recommendation = get_rec(my_profile)

    if recommendation is None:
        print("No recommendation found")
    else:
        print(recommendation)

