import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from GA import optimize_with_ga
from PSO import AdaptivePSO, repair_particle, fitness_function, calculate_pso_breakdown, optimize_with_pso

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

    # 4. Ejecutar PSO
    best_cost, best_position = optimize_with_pso(days, max_hours, lat, lon, categories, start_time)
    best_position = repair_particle(best_position, days, max_hours, start_time).reshape(days, -1)
    result = calculate_pso_breakdown(best_position, start_time, max_hours, categories)

    # 5. Reestructurar la salida
    days_list = []
    current_date = start_date

    for day in result["daily_breakdown"]:
        itinerary_pois = []
        for poi in day["schedule"]:
            itinerary_pois.append({
                "id": str(poi["id"]),
                "name": poi["name"],
                "latitude": str(poi["latitude"]),
                "longitude": str(poi["longitude"]),
                "address": poi["address"],
                "arrival_time": current_date.strftime("%d/%m/%Y") + f" {poi['arrival_time']} AM",
                "departure_time": current_date.strftime("%d/%m/%Y") + f" {poi['departure_time']} AM",
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

    return {
        "success": True,
        "itineraries": [{
            "name": "Itinerario 1",
            "startDate": request.startDate,
            "endDate": request.endDate,
            "days": days_list
        }]
    }