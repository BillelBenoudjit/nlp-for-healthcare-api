from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder

from server.database import (
     add_new_consultation,
     get_patients,
     get_patient,
     update_patient,
     delete_patient,
     add_consultation
)


from server.models.patient import (
     ConsultationSchema,
     PatientSchema
)

router = APIRouter()

'''
@router.post("/")
async def add_new_patient(patient: PatientSchema):
     patient = jsonable_encoder(patient)
     patient = await add_patient(patient)
     return patient
'''


@router.post("/")
async def add_consultation_to_patient(patient: PatientSchema):
     patient = jsonable_encoder(patient)
     new_patient = await add_new_consultation(patient)
     return new_patient


@router.get("/")
async def get_patients_data():
     patients = await get_patients()
     return patients


@router.get("/{id}")
async def get_patient_data(id: str):
     patient = await get_patient(id)
     return patient


@router.put("/{id}")
async def add_consultation_data(id: str, consultation: ConsultationSchema):
     consultation = {key: value for key, value in consultation.dict().items() if value is not None}
     patient_with_new_consultation = await add_consultation(id, consultation)
     return patient_with_new_consultation


