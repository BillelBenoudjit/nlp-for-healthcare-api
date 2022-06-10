import motor.motor_asyncio
from bson.objectid import ObjectId
from decouple import config
from fastapi import HTTPException

from server.models.patient import (
    ConsultationSchema,
    PatientSchema
)

MONGO_DETAILS = config("MONGO_DETAILS")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.health_nlp

patient_collection = database.get_collection("patients")


async def add_patient(patient):
    try:
        patient = await patient_collection.insert_one(patient)
        new_patient = await patient_collection.find_one({"_id": patient.inserted_id})
        return {"response": "User inserted: %s" %(new_patient)}
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def add_new_consultation(patient):
    try:
        checked_patient = await patient_collection.find_one({
            "firstName": patient["firstName"],
            "lastName": patient["lastName"],
            "age": patient["age"]
        })
        if checked_patient:
            id = str(checked_patient["_id"])
            if len(patient["consultations"]) > 0:
                print(patient["consultations"][0])
                consultation = patient["consultations"][0]
                patient_with_new_consultation = await add_consultation(id, consultation)
                return {"response": "Consultation added: %s" % (patient_with_new_consultation)}
            else:
                return {"response": "No consultation added: %s" % (checked_patient)}
        else:
            if len(patient["consultations"]) > 0:
                new_patient_data = {
                    "firstName": patient["firstName"],
                    "lastName": patient["lastName"],
                    "age": patient["age"],
                    "consultations": patient["consultations"]
                }
            else:
                new_patient_data = {
                    "firstName": patient["firstName"],
                    "lastName": patient["lastName"],
                    "age": patient["age"],
                    "consultations": []
                }
            patient = await patient_collection.insert_one(new_patient_data)
            new_patient = await patient_collection.find_one({"_id": patient.inserted_id})
            return {"response": "Patient inserted: %s" % (new_patient)}
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def get_patients():
    patients_list = []
    try:
        async for patient in patient_collection.find():
            patient["_id"] = str(patient["_id"])
            patients_list.append(patient)
        return patients_list
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def get_patient(id):
    try:
        patient = await patient_collection.find_one({"_id": ObjectId(id)})
        if patient:
            patient["_id"] = str(patient["_id"])
        return patient
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def update_patient(id, data):
    try:
        if len(data) < 1:
            return False
        patient = await patient_collection.find_one({"_id": ObjectId(id)})
        if patient:
            updated_patient = await patient_collection.update_one(
                {"_id": ObjectId(id)}, {"$set": data}
            )
            if updated_patient:
                return True
            return False
        return False
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def delete_patient(id):
    try:
        patient = await patient_collection.find_one({"_id": ObjectId(id)})
        if patient:
            await patient_collection.delete_one({"_id": ObjectId(id)})
            return True
        return False
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)


async def add_consultation(id, consultation):
    try:
        patient = await patient_collection.find_one({"_id": ObjectId(id)})
        if patient:
            updated_patient = await patient_collection.update_one(
                {"_id": ObjectId(id)},
                {"$push": {"consultations": consultation}}
            )
            if updated_patient:
                patient = await patient_collection.find_one({"_id": ObjectId(id)})
                patient["_id"] = str(patient["_id"])
                return patient
            return False
        return False
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)

