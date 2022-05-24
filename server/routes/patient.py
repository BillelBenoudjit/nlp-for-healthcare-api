from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder

from server.database import (
     add_patient,
     get_patients,
     get_patient,
     update_patient,
     delete_patient
)


from server.models.patient import (
     ConsultationSchema,
     PatientSchema
)

router = APIRouter()


@router.post("/")
async def add_new_patient(patient: PatientSchema):
     patient = jsonable_encoder(patient)
     patient = await add_patient(patient)
     return patient


@router.get("/")
async def get_patients_data():
     patients = await get_patients()
     return patients


@router.get("/{id}")
async def get_patient_data(id):
     patient = await get_patient(id)
     return patient
