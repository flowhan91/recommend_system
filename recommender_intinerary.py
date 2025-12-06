import math
import json
import numpy as np
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# Supabase Client Initialization
url = "https://gvtmqaakgyorevninbmt.supabase.co"
key = "sb_secret_WziBehEDoaGk4FOUGIYd5Q_qVJcs3Lj"
supabase: Client = create_client(url, key)

# Fetch all places once when the module is imported
response_all_places = (
    supabase.table("Place")
    .select("place_id, lat, lng, embedding")
    .execute()
)
all_places = response_all_places.data

# Haversine Distance Function
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

# Function to get nearby places
def get_nearby_places(input_data, all_places, haversine_distance, radius=1.5):
    user_lat = input_data['lat']
    user_lng = input_data['lng']

    nearby_places = []

    for place in all_places:
        if place['lat'] is not None and place['lng'] is not None:
            place_lat = place['lat']
            place_lng = place['lng']
            distance = haversine_distance(user_lat, user_lng, place_lat, place_lng)

            if distance <= radius:
                nearby_places.append(place)

    if not nearby_places:
        print(f"No places found within a {radius}km radius.")
        return []
    else:
        print(f"Found {len(nearby_places)} places within {radius}km.")
        return nearby_places

# Function to get user profile embeddings
def get_user_profile_embeddings(user_id, all_places, supabase):
    response_saved_places = (
        supabase.table("SavedPlaces")
        .select("place_id")
        .eq("user_id", user_id)
        .execute()
    )
    saved_place_ids = [item['place_id'] for item in response_saved_places.data]
    # print(f"Saved Place IDs: {saved_place_ids}")

    response_liked_places = (
        supabase.table("LikedPlaces")
        .select("place_id")
        .eq("user_id", user_id)
        .execute()
    )
    liked_place_ids = [item['place_id'] for item in response_liked_places.data]
    # print(f"Liked Place IDs: {liked_place_ids}")

    combined_place_ids = list(set(saved_place_ids + liked_place_ids))
    # print(f"Combined Unique Place IDs: {combined_place_ids}")

    user_profile_embeddings = []

    if not combined_place_ids:
        print("No saved or liked places found for the user.")
    else:
        for place in all_places:
            if place['place_id'] in combined_place_ids:
                if place['embedding'] is not None:
                    user_profile_embeddings.append(place['embedding'])

        if not user_profile_embeddings:
            print("No embeddings found for the saved/liked places.")

    # print(f"Collected {len(user_profile_embeddings)} embeddings for saved/liked places.")
    return user_profile_embeddings

# Function to calculate recommendation scores
def calculate_recommendation_scores(user_profile_embeddings, nearby_places):
    user_profile_np_embeddings = []

    if not user_profile_embeddings:
        print("No user profile embeddings found to generate recommendations.")
        average_user_embedding = None
    else:
        for emb_str in user_profile_embeddings:
            try:
                emb_list = json.loads(emb_str)
                user_profile_np_embeddings.append(np.array(emb_list))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse embedding string: {emb_str}")

        if user_profile_np_embeddings:
            average_user_embedding = np.mean(user_profile_np_embeddings, axis=0)
            # print("Successfully created average user embedding.")
        else:
            print("No valid user profile embeddings found after parsing.")
            average_user_embedding = None

    recommended_places = []

    if average_user_embedding is not None:
        for place in nearby_places:
            if place['embedding'] is not None:
                try:
                    place_emb_list = json.loads(place['embedding'])
                    place_np_embedding = np.array(place_emb_list)

                    user_emb_2d = average_user_embedding.reshape(1, -1)
                    place_emb_2d = place_np_embedding.reshape(1, -1)

                    similarity = cosine_similarity(user_emb_2d, place_emb_2d)[0][0]

                    recommended_places.append({
                        'place_id': place['place_id'],
                        'similarity_score': float(similarity),
                        #'place_details': place -> 안보냄
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse place embedding string for place_id {place['place_id']}")

        recommended_places.sort(key=lambda x: x['similarity_score'], reverse=True)

        # print("\nTop Recommended Places:")
        # for i, rec_place in enumerate(recommended_places):
        #     if i < 10: # Print top 10 for brevity
        #         print(f"Place ID: {rec_place['place_id']}, Similarity Score: {rec_place['similarity_score']:.4f}")
    else:
        print("Cannot calculate recommendations without an average user embedding.")
    
    return recommended_places

# Function to get itinerary places
def get_itinerary_places(itinerary_id, supabase):
    response_itinerary_places = (
        supabase.table("ItineraryPlace")
        .select("place_id")
        .eq("itinerary_id", itinerary_id)
        .execute()
    )
    itinerary_place_ids = [item['place_id'] for item in response_itinerary_places.data]
    # print(f"Itinerary Place IDs for itinerary {itinerary_id}: {itinerary_place_ids}")
    return itinerary_place_ids

# Function to exclude existing itinerary places
def exclude_existing_itinerary_places(recommended_places, itinerary_place_ids):
    filtered_recommendations = [
        place for place in recommended_places 
        if place['place_id'] not in itinerary_place_ids
    ]
    # print(f"Excluded {len(recommended_places) - len(filtered_recommendations)} places already in the itinerary.")
    # print(f"Final Recommended Places (after exclusion): {len(filtered_recommendations)}")
    return filtered_recommendations

# Main recommendation function
def recommend_next_place(input_data):
    # 1. Filter places by proximity
    # haversine_distance, all_places, supabase are now module-level variables/functions
    nearby_places = get_nearby_places(input_data, all_places, haversine_distance)
    if not nearby_places:
        return "1.5km내 장소가 없습니다"

    # 2. Generate user profile from saved/liked places
    user_profile_embeddings = get_user_profile_embeddings(input_data['user_id'], all_places, supabase)
    if not user_profile_embeddings:
        return "저장하거나 좋아요한 장소가 없습니다"

    # 3. Calculate recommendation scores
    recommended_places = calculate_recommendation_scores(user_profile_embeddings, nearby_places)
    if not recommended_places:
        return "추천할 장소를 찾지 못했습니다."

    # 4. Exclude existing itinerary places
    itinerary_place_ids = get_itinerary_places(input_data['itinerary_id'], supabase)
    final_recommendations = exclude_existing_itinerary_places(recommended_places, itinerary_place_ids)
    
    if not final_recommendations:
        return "추천할 장소가 모두 여정에 등록된 장소입니다."
    
    return final_recommendations

if __name__ == "__main__":
    # Example input data
    input_data = {
        "user_id" : "USR_a7F9kP12",
        "lat" : 35.149548 ,
        "lng" : 126.922151,
        "itinerary_id" : 108,
        "day_id" : 3
    }

    final_recommendations = recommend_next_place(input_data)

    print("\n--- Final Recommendations --- ")
    if isinstance(final_recommendations, str):
        print(final_recommendations)
    else:
        for i, rec_place in enumerate(final_recommendations):
            if i < 10: # Print top 10 for brevity
                print(f"Place ID: {rec_place['place_id']}, Score: {rec_place['similarity_score']:.4f}")
