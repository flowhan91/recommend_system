import math
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similaritys
import os
import supabase
from supabase import create_client

SUPABASE_URL = "https://gvtmqaakgyorevninbmt.supabase.co"
SUPABASE_KEY = "sb_secret_WziBehEDoaGk4FOUGIYd5Q_qVJcs3Lj"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# final_recommendations = get_recommendations(location_input)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Removed the custom cosine_similarity function as we will use sklearn's

def get_recommendations(location_input):
    user_id = location_input["user_id"]
    user_lat = location_input["lat"]
    user_lng = location_input["lng"]
    search_radius_km = 10
    #DEBUG
    print(f"[DEBUG] user_id={user_id}, user_lat={user_lat}, user_lng={user_lng}")

    # 1. Retrieve Nearby Places with valid embeddings
    response_all_places = (
        supabase.table("Place")
        .select("place_id, lat, lng, embedding")
        .execute()
    )
    
    all_places = response_all_places.data
    #DEBUG
    print("places collected:",len(all_places))

    nearby_places_data = []
    min_distance = float("inf")
    max_distance = 0
    sample_count = 0

    for place in all_places:
        place_lat = place['lat']
        place_lng = place['lng']
        distance = haversine_distance(user_lat, user_lng, place_lat, place_lng)

        # 거리 통계 기록
        if distance < min_distance:
            min_distance = distance
        if distance > max_distance:
            max_distance = distance

        # 앞의 몇 개만 샘플로 찍어보기
        if sample_count < 30:
            print(
                f"[DIST] place_id={place['place_id']} "
                f"place_lat={place_lat}, place_lng={place_lng}, "
                f"distance={distance}",
                flush=True
            )
            sample_count += 1

        if distance <= search_radius_km:
            print(f"[NEARBY] place_id={place['place_id']} distance={distance}", flush=True)

            embedding_str = place['embedding']
            if embedding_str:
                try:
                    place_embedding = [float(e) for e in json.loads(embedding_str)]
                    nearby_places_data.append({
                        "place_id": place['place_id'],
                        "embedding": place_embedding,
                        "lat": place['lat'],
                        "lng": place['lng'],
                        "distance": distance
                    })
                except json.JSONDecodeError as e:
                    print(f"[EMBEDDING ERROR] place_id={place['place_id']} error={e}", flush=True)
                    continue

    print(f"[SUMMARY] min_distance={min_distance}, max_distance={max_distance}", flush=True)
    print("nearby places collected:", len(nearby_places_data), flush=True)


    # 2. Retrieve User Profile Embedding 
    user_profile_embedding = None
    user_relevant_place_ids = []

    # Fetch saved places
    response_saved_places = (
        supabase.table("SavedPlaces")
        .select("place_id")
        .eq("user_id", user_id)
        .execute()
    )
    saved_place_ids = [item['place_id'] for item in response_saved_places.data]
    user_relevant_place_ids.extend(saved_place_ids)

    # Fetch liked places
    response_liked_places = (
        supabase.table("LikedPlaces")
        .select("place_id")
        .eq("user_id", user_id)
        .execute()
    )
    liked_place_ids = [item['place_id'] for item in response_liked_places.data]
    user_relevant_place_ids.extend(liked_place_ids)
    print("SavedPlaces:", saved_place_ids)
    print("LikedPlaces:", liked_place_ids)
    print("User relevant place IDs:", user_relevant_place_ids)
    
    # Remove duplicates and ensure there are relevant places
    user_relevant_place_ids = list(set(user_relevant_place_ids))

    if not user_relevant_place_ids:
        # If no liked/saved places, cannot create profile embedding
        print(f"User {user_id} has no saved or liked places to build a profile embedding.")

    # Fetch the embeddings for these relevant places from the 'Place' table
    response_user_places_embeddings = (
        supabase.table("Place")
        .select("place_id, embedding")
        .in_("place_id", user_relevant_place_ids)
        .execute()
    )
    user_places_embeddings_data = response_user_places_embeddings.data

    user_embeddings = []
    for place in user_places_embeddings_data:
        embedding_str = place['embedding']
        if embedding_str:
            try:
                embedding_list = json.loads(embedding_str)
                place_embedding = [float(e) for e in embedding_list]
                if place_embedding:
                    user_embeddings.append(place_embedding)
            except (json.JSONDecodeError, ValueError):
                continue
    
    #DEBUG
    print("User embeddings collected:", len(user_embeddings))
    if len(user_embeddings) > 0:
        print("Example user embedding:", user_embeddings[0][:5])  
    
    
    if user_embeddings:
        user_profile_embedding = np.mean(user_embeddings, axis=0).tolist()
    else:
        raise ValueError(f"No valid embeddings found for user {user_id}'s saved/liked places.")
    
    print("Nearby places with valid embeddings:", len(nearby_places_data))
    
    # 3. Calculate Recommendation Scores
    recommendations = []
    if user_profile_embedding and nearby_places_data:
        user_vec = np.array(user_profile_embedding).reshape(1, -1)
        for place in nearby_places_data:
            if place['embedding']:
                place_vec = np.array(place['embedding']).reshape(1, -1)
                # Calculate cosine similarity score (taste score) using sklearn
                cosine_score = sk_cosine_similaritys(user_vec, place_vec)[0][0]

                distance = place['distance']
                distance_score = 1 - (distance / search_radius_km)
                distance_score = max(0, min(1, distance_score))

                combined_rec_score = (0.7 * cosine_score) + (0.3 * distance_score)

                recommendations.append({
                    "place_id": place['place_id'],
                    "rec_score": combined_rec_score
                })

    # Sort recommendations by score in descending order
    recommendations.sort(key=lambda x: x['rec_score'], reverse=True)

    return recommendations




