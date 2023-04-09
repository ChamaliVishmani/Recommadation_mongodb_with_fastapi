import os
import pandas as pd

from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from typing import Optional, List
import motor.motor_asyncio

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


class StudentModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    email: EmailStr = Field(...)
    course: str = Field(...)
    gpa: float = Field(..., le=4.0)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Jane Doe",
                "email": "jdoe@example.com",
                "course": "Experiments, Science, and Fashion in Nanophotonics",
                "gpa": "3.0",
            }
        }


class UpdateStudentModel(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    course: Optional[str]
    gpa: Optional[float]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Jane Doe",
                "email": "jdoe@example.com",
                "course": "Experiments, Science, and Fashion in Nanophotonics",
                "gpa": "3.0",
            }
        }


# const { Schema, model } = require("mongoose");
# const Order = require("./order_model");
# const Food = require("./food_model");
#
# const orderItemsSchema = new Schema(
#   {
#     orderID: { type: Schema.Types.ObjectId, ref: "Order" },
#     food: { type: Schema.Types.ObjectId, ref: "Food" },
#     quanitity: { type: Number },
#     price: { type: Number },
#   },
#   { collection: "orderItemWithQuantities" }
# );
#
# module.exports = model("OrderItemWithQuantity", orderItemsSchema);

@app.post("/", response_description="Add new student", response_model=StudentModel)
async def create_student(student: StudentModel = Body(...)):
    student = jsonable_encoder(student)
    new_student = await db["students"].insert_one(student)
    created_student = await db["students"].find_one({"_id": new_student.inserted_id})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_student)


@app.get("/", response_description="List all students", response_model=List[StudentModel])
async def list_students():
    students = await db["students"].find().to_list(1000)
    return students


# @app.get(
#     "/{id}", response_description="Get a single student", response_model=StudentModel
# )
# async def show_student(id: str):
#     if (student := await db["students"].find_one({"_id": id})) is not None:
#         return student
#
#     raise HTTPException(status_code=404, detail=f"Student {id} not found")

@app.put("/{id}", response_description="Update a student", response_model=StudentModel)
async def update_student(id: str, student: UpdateStudentModel = Body(...)):
    student = {k: v for k, v in student.dict().items() if v is not None}

    if len(student) >= 1:
        update_result = await db["students"].update_one({"_id": id}, {"$set": student})

        if update_result.modified_count == 1:
            if (
                    updated_student := await db["students"].find_one({"_id": id})
            ) is not None:
                return updated_student

    if (existing_student := await db["students"].find_one({"_id": id})) is not None:
        return existing_student

    raise HTTPException(status_code=404, detail=f"Student {id} not found")


@app.delete("/{id}", response_description="Delete a student")
async def delete_student(id: str):
    delete_result = await db["students"].delete_one({"_id": id})

    if delete_result.deleted_count == 1:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    raise HTTPException(status_code=404, detail=f"Student {id} not found")


# const foodSchema = new Schema(
#   {
#     name: { type: String },
#     price: { type: Number },
#     description: { type: String },
#     image: { type: String },
#     category: { type: Schema.Types.ObjectId, ref: "FoodCatergory" }, //Get object id from foodCategory_model
#   },
#   { collection: "foods" }
# );
#
# module.exports = model("Food", foodSchema);

class FoodModel(BaseModel):
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


class UpdateFoodModel(BaseModel):
    name: Optional[str]
    price: Optional[float]
    description: Optional[str]
    image: Optional[str]
    category: Optional[str]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "name": "Fish and Chips",
                "price": "10.0",
                "description": "Fish and Chips",
                "image": "image",
                "category": "category",
            }
        }


# List all foods
@app.get("/foods", response_description="List all foods", response_model=List[FoodModel])
async def list_foods():
    foods = await db["foodDetails"].find().to_list(1000)
    return foods


# const orderItemsSchema = new Schema(
#   {
#     orderID: { type: Schema.Types.ObjectId, ref: "Order" },
#     food: { type: Schema.Types.ObjectId, ref: "Food" },
#     quanitity: { type: Number },
#     price: { type: Number },
#   },
#   { collection: "orderItemWithQuantities" }
# );
#
# module.exports = model("OrderItemWithQuantity", orderItemsSchema);


class OrderItemWithQuantity(BaseModel):
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


@app.get("/orderItemWithQuantities", response_description="List all order items with quantities",
         response_model=List[OrderItemWithQuantity])
async def list_order_items_with_quantities():
    order_items_with_quantities = await db["orderItemWithQuantities"].find().to_list(1000)
    return order_items_with_quantities


# const orderSchema = new Schema(
#   {
#     createDate: { type: String },
#     createTime: { type: String },
#     status: { type: String },
#     orderedBy: { type: Schema.Types.ObjectId, ref: "User" },
#     billValue: { type: String },
#     discount: { type: String },
#     orderType: { type: Schema.Types.ObjectId, ref: "OrderType" },
#     table: { type: Schema.Types.ObjectId, ref: "Table" },
#     handleBy: { type: Schema.Types.ObjectId, ref: "Employee" },
#   },
#   { collection: "orders" }
# );
#
# module.exports = model("Order", orderSchema);


# const { Schema, model } = require("mongoose");
#
# const userSchema = new Schema(
#   {
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     firstName: { type: String },
#     lastName: { type: String },
#     userName: { type: String },
#     email: { type: String },
#     dateOfBirth: { type: String },
#     mobileNumber: { type: String },
#     password: { type: String },
#   },
#   { collection: "users" }
# );
#
# module.exports = model("User", userSchema);

class User(BaseModel):
    userID: str
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


# const { Schema, model } = require("mongoose");
#
# const foodDetailsSchema = new Schema(
#   {
#     food_id: { type: Schema.Types.ObjectId, ref: "User" },
#     food_name: { type: String },
#     food_type: { type: String },
#     Ingredients: { type: String },
#   },
#   { collection: "foodDetails" }
# );

class FoodDetails(BaseModel):
    food_id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    food_name: str = Field(...)
    food_type: str = Field(...)
    cuisine: str = Field(...)
    Ingredients: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "food_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "food_name": "Pizza",
                "food_type": "Main",
                "cuisine": "Italian",
                "Ingredients": "Cheese, Tomato, Onion"
            }
        }


# const { Schema, model } = require("mongoose");
# const User = require("../models/user_model");
#
# const feedbackSchema = new Schema(
#   {
#     feedback: { type: String },
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     orderId: { type: Schema.Types.ObjectId, ref: "Order" },
#   },
#   { collection: "feedbacks" }
# );
#
# module.exports = model("Feedback", feedbackSchema);

class Feedback(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    feedback: str
    orderId: PyObjectId = Field(default_factory=PyObjectId)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "feedback": "Good",
                "orderId": "60f4d5c5b5f0f0e5e8b2b5c9"
            }
        }


@app.get("/feedbacks", response_description="List all feedbacks", response_model=List[Feedback])
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
    orderItemWithQuantities: List[OrderItemWithQuantity]

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
