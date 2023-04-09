import os
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
    foods = await db["foodDetails"].find().to_list(1000)
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
    food: str
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
    order_items_with_quantities = await db["orderItemWithQuantities"].find().to_list(1000)
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
    orders = await db["orders"].find().to_list(1000)
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
    users = await db["users"].find().to_list(1000)
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
    feedbacks = await db["feedbacks"].find().to_list(100)
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

@app.get("/aggregate", response_description="Aggregate data", response_model=List[OrderModel])
async def aggregate_data():
    orders = await db["orders"].find().to_list(1000)
    # orderItemWithQuantities = await db["orderItemWithQuantities"].find().to_list(1000)
    # foodDetails = await db["foodDetails"].find()
    # users = await db["users"].find()
    # feedbacks = await db["feedbacks"].find()

    print(orders)

    # # 1. Aggregate orders with orderItemWithQuantities
    # for order in orders:
    #     for orderItem in orderItemWithQuantities:
    #         if orderItem.orderId == order.id:
    #             order.orderItemWithQuantities.append(orderItem)
    #
    # # 2. for each orderItem get orderedBy and for that get dateOfBirth from users
    # for order in orders:
    #     for orderItem in order.orderItemWithQuantities:
    #         for user in users:
    #             if order.orderedBy == user.id:
    #                 orderItem.orderedBy = user
    #                 break
    #
    # # 3. for each orderItem get food_id, food_type, Ingredients from foodDetails by food_name
    # for order in orders:
    #     for orderItem in order.orderItemWithQuantities:
    #         for foodDetail in foodDetails:
    #             if orderItem.food == foodDetail.food:
    #                 orderItem.food_id = foodDetail.id
    #                 orderItem.food_type = foodDetail.food_type
    #                 orderItem.Ingredients = foodDetail.Ingredients
    #                 break
    #
    # # 4. for each orderItem get food_rating from feedbacks
    # for order in orders:
    #     for orderItem in order.orderItemWithQuantities:
    #         for feedback in feedbacks:
    #             if orderItem.id == feedback.orderId:
    #                 orderItem.food_rating = feedback.feedback
    #                 break
    #
    # # convert to dataframes
    # orders_df = pd.DataFrame(orders)
    # orderItemWithQuantities_df = pd.DataFrame(orderItemWithQuantities)
    # foodDetails_df = pd.DataFrame(foodDetails)
    # users_df = pd.DataFrame(users)
    # feedbacks_df = pd.DataFrame(feedbacks)

    # merge dataframes
    # df = pd.merge(orders_df, orderItemWithQuantities_df, left_on='id', right_on='orderId')
    # df = pd.merge(df, foodDetails_df, left_on='food', right_on='food')
    # df = pd.merge(df, users_df, left_on='orderedBy', right_on='id')
    # df = pd.merge(df, feedbacks_df, left_on='id', right_on='orderId')
    #

    # print(orders_df)
    return orders


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


def process_data():
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

    # print list of columns in each dataframe
    print("orderItemWithQuantities_df: ", orderItemWithQuantities_df.columns)
    print("orders_df: ", orders_df.columns)
    print("foodDetails_df: ", foodDetails_df.columns)
    print("users_df: ", users_df.columns)
    print("feedbacks_df: ", feedbacks_df.columns)


    # add orders_df to orderWithQuantities_df by id to orderID
    df = pd.merge(orderItemWithQuantities_df, orders_df, left_on='orderID', right_on='_id')


    # save to csv
    df.to_csv(save_path + "aggregate.csv", index=False)

    print("\n\nName of the file: aggregate")
    print(df.head())


if __name__ == "__main__":
    # call_api()
    process_data()
