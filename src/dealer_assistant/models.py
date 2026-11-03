"""Data model of one car in the stock. Values are validated when the stock is loaded."""

from typing import Literal

from pydantic import BaseModel, Field

Currency = Literal["ARS", "USD"]
Category = Literal["sedan", "hatchback", "suv", "pickup", "van", "truck", "motorcycle"]
Transmission = Literal["MT", "AT", "CVT"]
Fuel = Literal["gasoline", "diesel", "cng", "hybrid", "electric"]
Status = Literal["available", "reserved", "sold"]
Condition = Literal["used", "new"]


class Car(BaseModel):
    id: str = Field(min_length=1)
    make: str
    model: str
    trim: str
    year: int = Field(ge=1990, le=2100)
    km: int = Field(ge=0)
    price: int = Field(gt=0)
    currency: Currency
    category: Category
    transmission: Transmission
    fuel: Fuel
    status: Status
    condition: Condition
    immediate_delivery: bool
