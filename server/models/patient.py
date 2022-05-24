from typing import Optional, List, Union

from pydantic import BaseModel, Field


class ConsultationSchema(BaseModel):
    entities: list = []
    time: Optional[str]
    date: Optional[str]


class PatientSchema(BaseModel):
    firstName: str = Field(...)
    lastName: str = Field(...)
    age: str = Field(...)
    consultations:  List[ConsultationSchema]

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "firstName": "Ben",
            "lastName": "Bill",
            "age": "20",
            "consultations": [
                {
                    "entities": [],
                    "time": "",
                    "date": ""
                }
            ]
        }

