import logging
from fastapi import FastAPI, HTTPException
import datetime
from pydantic import BaseModel
from typing import List
import numpy as np
from PSO import (
    generate_three_itineraries  # PSO
)
from GA import (
    generate_three_itineraries_with_ga  # AG
)

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

class ItineraryRequest(BaseModel):
    startDate: str
    startTime: str
    endDate: str
    endTime: str
    location: dict
    categories: List[str]

# Endpoint para PSO
@app.post("/generate-itinerary/")
def generate_itinerary(request: ItineraryRequest):
    return generate_itinerary_core(request, use_ga=False)

# Endpoint para AG
@app.post("/generate-itinerary-ga/")
def generate_itinerary_ga(request: ItineraryRequest):
    return generate_itinerary_core(request, use_ga=True)

# Función reutilizada para ambos algoritmos
def generate_itinerary_core(request: ItineraryRequest, use_ga: bool = False):
    try:
        # Log del JSON de entrada
        logging.info(f"Solicitud recibida: {request.json()}")

        # 1. Calcular días
        start_date = datetime.datetime.strptime(request.startDate, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(request.endDate, "%Y-%m-%d")
        days = (end_date - start_date).days + 1
        logging.info(f"Días calculados: {days}")

        # 2. Calcular horas de turismo por día
        start_time = datetime.datetime.strptime(request.startTime, "%H:%M")
        end_time = datetime.datetime.strptime(request.endTime, "%H:%M")
        max_hours = (end_time - start_time).seconds / 3600
        logging.info(f"Horas máximas por día: {max_hours}")

        # 3. Extraer ubicación y categorías
        lat = request.location["latitude"]
        lon = request.location["longitude"]
        categories = request.categories
        logging.info(f"Ubicación: lat={lat}, lon={lon}, Categorías: {categories}")

        # 4. Ejecutar algoritmo correspondiente
        if use_ga:
            logging.info("Ejecutando algoritmo AG...")
            itinerarios = generate_three_itineraries_with_ga(days, max_hours, lat, lon, categories, start_time)
        else:
            logging.info("Ejecutando algoritmo PSO...")
            itinerarios = generate_three_itineraries(days, max_hours, lat, lon, categories, start_time)

        response = []
        for idx, result in enumerate(itinerarios, start=1):
            # Omitir itinerarios si algún día está vacío
            if any(len(day["schedule"]) == 0 for day in result["daily_breakdown"]):
                logging.warning(f"Itinerario {idx} omitido debido a días vacíos.")
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
            logging.error("No se pudo generar ningún itinerario completo con los criterios proporcionados.")
            raise HTTPException(
                status_code=400,
                detail="No se pudo generar ningún itinerario completo con los criterios proporcionados."
            )

        logging.info("Itinerarios generados exitosamente.")
        return {
            "success": True,
            "itineraries": response
        }

    except ValueError as e:
        logging.error(f"Error en la solicitud: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
