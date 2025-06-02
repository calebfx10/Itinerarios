import random
import numpy as np
from deap import base, creator, tools, algorithms
import time
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta, datetime






# Definir los pesos externamente
weights = {
    'transit_weight': 0.7,
    'rating_weight': 0.2,
    'preference_weight': 0.1
}



POPULATION_SIZE = 500
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.9
GENERATIONS = 100



MAX_ATTEMPTS = 100


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def calculate_total_time(day, start_time):
    if not day:
        return 0
    
    total_time = 0
    current_time = start_time

    for i in range(len(day)):
        poi = day[i]

        # Añadir duración del POI
        total_time += POIs[poi]['duration']
        current_time += timedelta(hours=POIs[poi]['duration'])
        
        # Añadir tiempo de tránsito si no es el último POI del día
        if i < len(day) - 1:
            transit = transit_time.get((day[i], day[i + 1]), 0.5)
            total_time += transit
            current_time += timedelta(hours=transit)
    
    return total_time


def evaluate(individual, preferred_categories, weights, start_time):
    total_transit_time = 0
    total_rating = 0
    preferred_matches = 0
    penalty = 0

    for day in individual:
        day_time = calculate_total_time(day, start_time)
        if day_time > MAX_HOURS_PER_DAY:
            penalty += (day_time - MAX_HOURS_PER_DAY) * 2
        elif day_time < MAX_HOURS_PER_DAY:
            penalty += (MAX_HOURS_PER_DAY - day_time) * 0.5

        for i, poi in enumerate(day):
            total_rating += POIs[poi]['rating']
            poi_cats = [c.strip() for c in POIs[poi]['category'].split(',')]
            if any(c in preferred_categories for c in poi_cats):
                preferred_matches += 1


            if i < len(day) - 1:
                transit = transit_time.get((day[i], day[i + 1]), 0.5)
                total_transit_time += transit

    fitness = (weights['transit_weight'] * -total_transit_time) + \
              (weights['rating_weight'] * total_rating) + \
              (weights['preference_weight'] * preferred_matches * 10) - penalty

    return fitness,


def create_individual():
    individual = []
    for _ in range(DAYS):
        day = []
        total_time = 0
        attempts = 0
        while total_time < MAX_HOURS_PER_DAY and attempts < MAX_ATTEMPTS:
            poi = random.choice(list(POIs.keys()))
            if poi not in day:
                new_time = total_time + POIs[poi]['duration']
                if len(day) > 0:
                    new_time += transit_time.get((day[-1], poi), 0.5)
                if new_time <= MAX_HOURS_PER_DAY:
                    day.append(poi)
                    total_time = new_time
                else:
                    attempts += 1
            else:
                attempts += 1
        individual.append(day)
    return individual

def mutate_day(day):
    if random.random() < MUTATION_RATE:
        if len(day) > 0:
            if random.random() < 0.5 and len(day) > 1:
                day.pop(random.randint(0, len(day) - 1))
            else:
                index = random.randint(0, len(day) - 1)
                new_poi = random.choice(list(POIs.keys()))
                while new_poi in day:
                    new_poi = random.choice(list(POIs.keys()))
                day[index] = new_poi
    return day

def mutate(individual):
    mutated_individual = [mutate_day(day) for day in individual]
    return creator.Individual(mutated_individual),

def crossover(ind1, ind2):
    if random.random() < CROSSOVER_RATE:
        cut_point = random.randint(1, DAYS - 1)
        child1 = creator.Individual(ind1[:cut_point] + ind2[cut_point:])
        child2 = creator.Individual(ind2[:cut_point] + ind1[cut_point:])
        return child1, child2
    return ind1, ind2

def repair(individual):
    repaired_individual = []
    visited_pois = set()

    for day in individual:
        repaired_day = []
        for poi in day:
            if isinstance(poi, int) and poi in POIs:
                if poi not in visited_pois:
                    repaired_day.append(poi)
                    visited_pois.add(poi)
                else:
                    available_pois = [p for p in POIs.keys() if p not in visited_pois and p not in repaired_day]
                    if available_pois:
                        new_poi = random.choice(available_pois)
                        repaired_day.append(new_poi)
                        visited_pois.add(new_poi)
        
        repaired_individual.append(repaired_day)

    return creator.Individual(repaired_individual)

def calculate_fitness_breakdown(individual, weights, start_time, preferred_categories):
    total_transit_time = 0
    total_rating = 0
    categories_visited = set()
    penalty = 0
    daily_breakdown = []
    
    for day_index, day in enumerate(individual, 1):
        day_time = calculate_total_time(day, start_time)
        current_time = start_time
        day_transit = 0
        day_rating = 0
        day_categories = set()
        day_penalty = 0
        schedule = []
        
        if day_time > MAX_HOURS_PER_DAY:
            day_penalty = (day_time - MAX_HOURS_PER_DAY) * 1.5
        elif day_time < MAX_HOURS_PER_DAY:
            day_penalty = (MAX_HOURS_PER_DAY - day_time) * 1.5
            
        for i, poi in enumerate(day):
            # Considerar el tiempo de tránsito antes de iniciar en cada POI (excepto el primero)
            if i > 0:
                transit_duration = transit_time.get((day[i-1], day[i]), 0.5)
                current_time += timedelta(hours=transit_duration)
                day_transit += transit_duration
            
            # Calcular la hora de inicio y fin del POI actual
            poi_start_time = current_time
            poi_duration = POIs[poi]['duration']
            current_time += timedelta(hours=poi_duration)
            poi_end_time = current_time
            
            schedule.append({
                'id': int(poi),
                'name': POIs[poi]['name'],
                'latitude': float(POIs[poi].get('latitude', 0)),  # Si aún no guardas esto, añádelo en POIs
                'longitude': float(POIs[poi].get('longitude', 0)),
                'address': POIs[poi].get('address', 'Dirección no disponible'),
                'arrival_time': (current_time - timedelta(hours=poi_duration)).strftime("%H:%M"),
                'departure_time': current_time.strftime("%H:%M"),
                'category': POIs[poi]['category'],
                'rating': float(POIs[poi]['rating']),
                'photos': POIs[poi].get('photos', []),
                'description': POIs[poi].get('description', 'Sin descripción disponible')
            })
            day_rating += POIs[poi]['rating']
            day_categories.add(POIs[poi]['category'])
        
        total_transit_time += day_transit
        total_rating += day_rating
        categories_visited.update(day_categories)
        penalty += day_penalty
        
        daily_breakdown.append({
            'day': day_index,
            'schedule': schedule,
            'total_time': day_time,
            'transit_time': day_transit,
            'rating': day_rating,
            'categories': list(day_categories),
            'penalization': day_penalty
        })
    
    transit_component = weights['transit_weight'] * -total_transit_time
    rating_component = weights['rating_weight'] * total_rating
    preference_component = weights['preference_weight'] * len(categories_visited) * 10
    
    final_fitness = transit_component + rating_component + preference_component - penalty
    
    return {
        'daily_breakdown': daily_breakdown,
        'total_transit_time': total_transit_time,
        'total_rating': total_rating,
        'preferred_categories': len([cat for cat in categories_visited if cat in preferred_categories]),
        'total_penalization': penalty
    }

def run_ga(weights, preferred_categories, start_time):
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate(ind, preferred_categories, weights, start_time))
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)
    
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
        offspring = [repair(ind) for ind in offspring]
        
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
            
        population = toolbox.select(offspring + population, k=len(population))

    return tools.selBest(population, k=1)[0]

def optimize_with_ga(days, max_hours, lat, lon, categories, start_time, exclude_pois=None):
    global POIs, transit_time, DAYS, MAX_HOURS_PER_DAY

    # Actualizar parámetros globales
    DAYS = days
    MAX_HOURS_PER_DAY = max_hours

    # Conectar a base de datos
    engine = create_engine('postgresql://postgres:itinerarios12345@db.dfbrazdcvlwnwlmlaxpm.supabase.co:5432/postgres')
    categoria_sql_like = [f"%{c}%" for c in categories]

    sql = """
        SELECT id, name, rating, category, latitude, longitude, address, description
        FROM pois 
        WHERE ST_DWithin(geom, ST_MakePoint(%s, %s)::geography, 8000)
          AND category ILIKE ANY (%s)
    """
    params = (lon, lat, categoria_sql_like)

    if exclude_pois:
        sql += " AND id != ALL(%s)"
        params += ([*exclude_pois],)


    pois_df = pd.read_sql(sql, engine, params=params)

    valid_ids = pois_df['id'].tolist()

    photos_df = pd.read_sql("""
        SELECT photo_url, poi_id
        FROM poi_photo
        WHERE poi_id = ANY(%s)
    """, engine, params=(valid_ids,))

    poi_photos = {}
    for poi_id, group in photos_df.groupby('poi_id'):
        poi_photos[poi_id] = group['photo_url'].tolist()

    transit_df = pd.read_sql("""
        WITH near_pois AS (
            SELECT unnest(%s::int[]) AS id
        )
        SELECT poi_start, poi_end, transit_time
        FROM poi_distances
        WHERE poi_start IN (SELECT id FROM near_pois)
          AND poi_end IN (SELECT id FROM near_pois)
    """, engine, params=(valid_ids,))
    
    engine.dispose()

    # Actualizar POIs y transit_time globales
    POIs = {
        row['id']: {
            'name': row['name'],
            'duration': 2,
            'rating': row['rating'],
            'category': row['category'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'address': row.get('address', 'Dirección no disponible'),
            'description': row.get('description', 'Sin descripción disponible'),
            'photos': poi_photos.get(row['id'], [])

        }
        for _, row in pois_df.iterrows()
    }

    transit_time = {
        (row['poi_start'], row['poi_end']): row['transit_time']
        for _, row in transit_df.iterrows()
    }

    if not POIs:
        raise ValueError("No se encontraron POIs válidos para los parámetros dados.")

    best_individual = run_ga(weights, categories, start_time)
    breakdown = calculate_fitness_breakdown(best_individual, weights, start_time, categories)
    return breakdown

def generate_three_itineraries_with_ga(days, max_hours, lat, lon, categories, start_time):
    all_itineraries = []
    used_pois = set()

    for i in range(3):
        print(f"[GA] Generando itinerario {i + 1}...")

        breakdown = optimize_with_ga(
            days=days,
            max_hours=max_hours,
            lat=lat,
            lon=lon,
            categories=categories,
            start_time=start_time,
            exclude_pois=list(used_pois) if used_pois else None
        )

        all_itineraries.append(breakdown)

        for day in breakdown["daily_breakdown"]:
            if day["schedule"]:
                poi_ids = [poi["id"] for poi in day["schedule"]]
                selected_id = random.choice(poi_ids)
                used_pois.add(selected_id)

    return all_itineraries
