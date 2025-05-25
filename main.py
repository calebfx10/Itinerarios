from fastapi import FastAPI, HTTPException
import datetime
from pydantic import BaseModel
from typing import List
import numpy as np
from PSO import (
    AdaptivePSO,
    repair_particle,
    fitness_function,
    calculate_pso_breakdown,
    optimize_with_pso,
    generate_three_itineraries
)

app = FastAPI()

class ItineraryRequest(BaseModel):
    startDate: str
    startTime: str
    endDate: str
    endTime: str
    location: dict
    categories: List[str]

@app.post("/generate-itinerary/")
def generate_itinerary(request: ItineraryRequest):
    try:
        # 1. Calcular días
        start_date = datetime.datetime.strptime(request.startDate, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(request.endDate, "%Y-%m-%d")
        days = (end_date - start_date).days + 1

        # 2. Calcular horas de turismo por día
        start_time = datetime.datetime.strptime(request.startTime, "%H:%M")
        end_time = datetime.datetime.strptime(request.endTime, "%H:%M")
        max_hours = (end_time - start_time).seconds / 3600

        # 3. Extraer ubicación y categorías
        lat = request.location["latitude"]
        lon = request.location["longitude"]
        categories = request.categories

        # 4. Ejecutar PSO 3 veces para obtener 3 itinerarios distintos
        itinerarios = generate_three_itineraries(days, max_hours, lat, lon, categories, start_time)

        response = []
        for idx, result in enumerate(itinerarios, start=1):
            # Omitir itinerarios si algún día está vacío
            if any(len(day["schedule"]) == 0 for day in result["daily_breakdown"]):
                continue

            days_list = []
            current_date = start_date
            for day in result["daily_breakdown"]:
                itinerary_pois = []
                for poi in day["schedule"]:
                    itinerary_pois.append({
                        "id": poi["id"],
                        "name": poi["name"],
                        "latitude": str(poi["latitude"]),
                        "longitude": str(poi["longitude"]),
                        "address": poi["address"],
                        "arrival_time": current_date.strftime("%d/%m/%Y") + " " + datetime.datetime.strptime(poi['arrival_time'], "%H:%M").strftime("%I:%M %p"),
                        "departure_time": current_date.strftime("%d/%m/%Y") + " " + datetime.datetime.strptime(poi['departure_time'], "%H:%M").strftime("%I:%M %p"),
                        "category": poi["category"],
                        "rating": poi["rating"],
                        "photos": poi["photos"],
                        "description": poi["description"]
                    })
                days_list.append({
                    "number": day["day"],
                    "itinerary_pois": itinerary_pois
                })
                current_date += datetime.timedelta(days=1)

            response.append({
                "name": f"Itinerario {idx}",
                "startDate": request.startDate,
                "endDate": request.endDate,
                "days": days_list
            })

        if not response:
            raise HTTPException(
                status_code=400,
                detail="No se pudo generar ningún itinerario completo con los criterios proporcionados."
            )

        return {
            "success": True,
            "itineraries": response
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
