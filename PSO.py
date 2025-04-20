from pyswarms.single import GlobalBestPSO
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sqlalchemy import create_engine
from pyswarms.single.global_best import GlobalBestPSO
import os
from pyswarms.single.global_best import GlobalBestPSO
import numpy as np

class AdaptivePSO(GlobalBestPSO):
    def __init__(self, n_particles, dimensions, w_bounds, c1_bounds, c2_bounds, **kwargs):
        super().__init__(n_particles=n_particles, dimensions=dimensions, options={}, **kwargs)
        self.w_bounds = w_bounds
        self.c1_bounds = c1_bounds
        self.c2_bounds = c2_bounds

    def update_hyperparameters(self, i, iters):
        w = self.w_bounds[0] - (i / iters) * (self.w_bounds[0] - self.w_bounds[1])
        c1 = self.c1_bounds[0] - (i / iters) * (self.c1_bounds[0] - self.c1_bounds[1])
        c2 = self.c2_bounds[0] + (i / iters) * (self.c2_bounds[1] - self.c2_bounds[0])
        return w, c1, c2

    def optimize(self, objective_func, iters, **kwargs):
        # Evaluar inicialmente para inicializar best_pos y best_cost
        self.swarm.current_cost = objective_func(self.swarm.position)
        self.swarm.pbest_pos = self.swarm.position.copy()
        self.swarm.pbest_cost = self.swarm.current_cost.copy()
        self.swarm.best_pos = self.swarm.pbest_pos[np.argmin(self.swarm.pbest_cost)].copy()
        self.swarm.best_cost = np.min(self.swarm.pbest_cost)

        for i in range(iters):
            w, c1, c2 = self.update_hyperparameters(i, iters)

            # Actualizar velocidad
            cognitive = c1 * np.random.uniform(size=self.swarm.position.shape) * (self.swarm.pbest_pos - self.swarm.position)
            social = c2 * np.random.uniform(size=self.swarm.position.shape) * (self.swarm.best_pos - self.swarm.position)
            self.swarm.velocity = w * self.swarm.velocity + cognitive + social

            # Actualizar position
            self.swarm.position += self.swarm.velocity

            # Evaluar nueva position
            self.swarm.current_cost = objective_func(self.swarm.position)
            mask = self.swarm.current_cost < self.swarm.pbest_cost
            self.swarm.pbest_pos[mask] = self.swarm.position[mask]
            self.swarm.pbest_cost[mask] = self.swarm.current_cost[mask]

            # Actualizar mejor global
            if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
                self.swarm.best_pos = self.swarm.pbest_pos[np.argmin(self.swarm.pbest_cost)].copy()
                self.swarm.best_cost = np.min(self.swarm.pbest_cost)

        return self.swarm.best_cost, self.swarm.best_pos


# Reparación de partículas con lógica mejorada
def repair_particle(particle, days, max_hours, start_time):
    particle = (particle * len(POIs)).astype(int)
    visited = set()
    repaired_itinerary = []

    for day_idx, day in enumerate(particle.reshape(days, -1), start=1):
        repaired_day = []
        day_time = 0
        current_time = start_time

        for poi in day:
            if poi in visited or poi not in POIs:
                continue


            poi_duration = POIs[poi]['duration']
            if day_time + poi_duration > max_hours:
                continue

            visited.add(poi)
            repaired_day.append(poi)
            day_time += poi_duration
            current_time += timedelta(hours=poi_duration)

            if len(repaired_day) > 1:
                transit = transit_time.get((repaired_day[-2], poi), 0.5)
                if day_time + transit > max_hours:
                    break
                day_time += transit
                current_time += timedelta(hours=transit)

        repaired_itinerary.extend(repaired_day + [0] * (len(day) - len(repaired_day)))

    return np.array(repaired_itinerary)

# Función de fitness con penalizaciones
def fitness_function(X, days, max_hours, categories, start_time):
    fitness = []
    for particle_idx, particle in enumerate(X):
        particle = repair_particle(particle, days, max_hours,start_time)
        itinerary = particle.reshape(days, -1)
        total_transit_time = 0
        total_rating = 0 
        preferred_matches = 0 
        penalty = 0 
        visited = set() 

        for day_num, day in enumerate(itinerary, 1):
            current_time = start_time
            day_transit = 0
            day_rating = 0
            day_time = 0
            valid_pois = [poi for poi in day if poi != 0 and poi in POIs]

            for i, poi in enumerate(valid_pois):
                if poi in visited:
                    penalty += 0.1
                    continue


                if i > 0:
                    prev_poi = valid_pois[i-1]
                    transit = transit_time.get((prev_poi, poi), 0.5)
                    if day_time + transit > max_hours:
                        penalty += (day_time + transit - max_hours) * 1.5
                    day_transit += transit
                    day_time += transit
                    current_time += timedelta(hours=transit)

                poi_duration = POIs[poi]['duration']
                if day_time + poi_duration > max_hours:
                    penalty += (day_time + poi_duration - max_hours) * 1.5

                visited.add(poi)
                day_rating += POIs[poi]['rating']
                if any(c in POIs[poi]['category'] for c in categories):
                    preferred_matches += 1


                current_time += timedelta(hours=poi_duration)
                day_time += poi_duration

            if day_time < max_hours:
                penalty += (max_hours - day_time) * 0.5

            total_transit_time += day_transit
            total_rating += day_rating

        fitness_value = (
            total_transit_time * 70 -
            total_rating * 30 -
            preferred_matches * 10 +
            penalty
        )

        fitness.append(fitness_value)

    return np.array(fitness)

# Función para calcular el desglose del itinerario en PSO
def calculate_pso_breakdown(itinerary, start_time, max_hours, preferred_categories):
    total_transit_time = 0
    total_rating = 0
    categories_visited = set()
    penalty = 0
    daily_breakdown = []

    for day_idx, day in enumerate(itinerary, 1):
        day_time = 0
        current_time = start_time
        day_transit = 0
        day_rating = 0
        day_categories = set()
        day_penalty = 0
        schedule = []

        valid_pois = [poi for poi in day if poi != 0 and poi in POIs]

        for i, poi in enumerate(valid_pois):


            # Añadir tiempo de tránsito desde el POI anterior
            if i > 0:
                prev_poi = valid_pois[i - 1]
                transit = transit_time.get((prev_poi, poi), 0.5)
                day_transit += transit
                current_time += timedelta(hours=transit)

            # Añadir duración del POI
            poi_duration = POIs[poi]['duration']
            current_time += timedelta(hours=poi_duration)
            day_time += poi_duration

            # Añadir a la programación del día
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
                'photos': ["https://example.com/photo.jpg"],  # Hardcode
                'description': POIs[poi].get('description', 'Sin descripción disponible')
            })


            # Actualizar métricas
            day_rating += POIs[poi]['rating']
            for cat in POIs[poi]['category'].split(','):
               day_categories.add(cat.strip())

        # Calcular el tiempo total del día (diferencia entre inicio y fin)
        if schedule:
            first_poi_start = datetime.strptime(schedule[0]['arrival_time'], "%H:%M")
            last_poi_end = datetime.strptime(schedule[-1]['departure_time'], "%H:%M")
            total_day_time = (last_poi_end - first_poi_start).seconds / 3600
        else:
            total_day_time = 0

        # Penalización por exceder o no alcanzar el tiempo máximo por día
        if total_day_time > max_hours:
            # Penalización por exceder el tiempo máximo
            day_penalty += (total_day_time - max_hours) * 1.5
        elif total_day_time < max_hours:
            # Penalización por no completar el tiempo mínimo
            day_penalty += (max_hours - total_day_time) * 0.5  # Ajusta el factor según sea necesario

        # Actualizar métricas totales
        total_transit_time += day_transit
        total_rating += day_rating
        categories_visited.update(day_categories)
        penalty += day_penalty

        # Añadir desglose del día
        daily_breakdown.append({
            'day': int(day_idx),
            'schedule': schedule,
            'total_time': total_day_time,  # Usar el tiempo total calculado
            'transit_time': day_transit,
            'rating': day_rating,
            'categories': list(day_categories),
            'penalization': day_penalty
        })

    return {
        'daily_breakdown': daily_breakdown,
        'total_transit_time': total_transit_time,
        'total_rating': total_rating,
        'preferred_categories': len([cat for cat in categories_visited if cat in preferred_categories]),
        'total_penalization': penalty
    }

def optimize_with_pso(days, max_hours, lat, lon, categories, start_time):
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL)

    categoria_sql_like = [f"%{c}%" for c in categories]

    pois_df = pd.read_sql("""
        SELECT id, name, rating, category, latitude, longitude, address, description
        FROM pois 
        WHERE ST_DWithin(geom, ST_MakePoint(%s, %s)::geography, 6000)
          AND category ILIKE ANY (%s)
    """, engine, params=(lon, lat, categoria_sql_like))

    valid_ids = pois_df['id'].tolist()

    transit_time_df = pd.read_sql("""
        WITH near_pois AS (
            SELECT unnest(%s::int[]) AS id
        )
        SELECT poi_start, poi_end, transit_time
        FROM poi_distances
        WHERE poi_start IN (SELECT id FROM near_pois)
          AND poi_end IN (SELECT id FROM near_pois)
    """, engine, params=(valid_ids,))
    engine.dispose()

    global POIs, transit_time
    POIs = {
        row['id']: {
            'name': row['name'],
            'duration': 2,
            'rating': row['rating'],
            'category': row['category'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'address': row.get('address', 'Dirección no disponible'),
            'description': row.get('description', 'Sin descripción disponible')
        }
        for _, row in pois_df.iterrows()
    }


    transit_time = {
        (row['poi_start'], row['poi_end']): row['transit_time']
        for _, row in transit_time_df.iterrows()
    }

    dimensions = len(POIs) * days * 2

    optimizer = AdaptivePSO(
        n_particles=50,
        dimensions=dimensions,
        w_bounds=(0.9, 0.4),
        c1_bounds=(2.5, 1.5),
        c2_bounds=(0.5, 2.0)
    )

    fitness_fn = lambda X: fitness_function(X, days, max_hours, categories, start_time)
    best_cost, best_pos = optimizer.optimize(fitness_fn, iters=100)
    return best_cost, best_pos