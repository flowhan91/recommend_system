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
# 서버 시작 시 한 번 로드됩니다.
response_all_places = (
    supabase.table("Place")
    .select("place_id, lat, lng, embedding")
    .execute()
)
all_places = response_all_places.data

# ---------------------------------------------------------
# [New Function] 추천 기준 위치(Anchor Location) 결정 로직
# ---------------------------------------------------------
def determine_anchor_location(itinerary_id, day_id, current_lat, current_lng, supabase):
    """
    추천의 중심이 되는 위치(Anchor)를 결정합니다.
    1순위: 해당 Day의 마지막 장소
    2순위: 해당 Itinerary 전체의 마지막 장소
    3순위: 유저의 현재 위치 (Input으로 들어온 값)
    """
    target_place_id = None
    
    # 1. 해당 날(day_id)에 장소가 있는지 확인
    day_places = (
        supabase.table("ItineraryPlace")
        .select("place_id")
        .eq("itinerary_id", itinerary_id)
        .eq("day_id", day_id)
        .execute()
    )
    
    if day_places.data and len(day_places.data) > 0:
        target_place_id = day_places.data[-1]['place_id'] # 마지막 장소
        print(f"Anchor found in Day {day_id}: Place ID {target_place_id}")
    else:
        # 2. 해당 날은 비었지만, 여정 전체에 장소가 있는지 확인
        itin_places = (
            supabase.table("ItineraryPlace")
            .select("place_id")
            .eq("itinerary_id", itinerary_id)
            .execute()
        )
        if itin_places.data and len(itin_places.data) > 0:
            target_place_id = itin_places.data[-1]['place_id']
            print(f"Anchor found in Itinerary {itinerary_id}: Place ID {target_place_id}")

    # 기준 장소가 정해졌으면 실제 좌표(lat, lng) 조회
    if target_place_id:
        place_info = (
            supabase.table("Place")
            .select("lat, lng")
            .eq("place_id", target_place_id)
            .single()
            .execute()
        )
        if place_info.data:
            return place_info.data['lat'], place_info.data['lng']
    
    # 3. 아무것도 없으면 원래 위치 반환
    print("[Log] No anchor found in itinerary. Using user's current location.")
    return current_lat, current_lng


# ---------------------------------------------------------
# Existing Helper Functions
# ---------------------------------------------------------

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
        # print(f"No places found within a {radius}km radius.")
        return []
    else:
        # print(f"Found {len(nearby_places)} places within {radius}km.")
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

    response_liked_places = (
        supabase.table("LikedPlaces")
        .select("place_id")
        .eq("user_id", user_id)
        .execute()
    )
    liked_place_ids = [item['place_id'] for item in response_liked_places.data]

    combined_place_ids = list(set(saved_place_ids + liked_place_ids))

    user_profile_embeddings = []

    if not combined_place_ids:
        # print("No saved or liked places found for the user.")
        pass
    else:
        for place in all_places:
            if place['place_id'] in combined_place_ids:
                if place['embedding'] is not None:
                    user_profile_embeddings.append(place['embedding'])

        if not user_profile_embeddings:
            print("No embeddings found for the saved/liked places.")
            pass
        print("total places with embeddings for user profile:", len(user_profile_embeddings) )
    return user_profile_embeddings

# Function to calculate recommendation scores
# 기존 calculate_recommendation_scores 함수를 이걸로 교체하세요

def calculate_recommendation_scores(user_profile_embeddings, nearby_places):
    print(f"\n[Debug] Calculating scores... Nearby places count: {len(nearby_places)}")
    
    user_profile_np_embeddings = []

    if not user_profile_embeddings:
        print("[Debug] No user profile embeddings found.")
        average_user_embedding = None
    else:
        for i, emb_str in enumerate(user_profile_embeddings):
            try:
                emb_list = json.loads(emb_str)
                user_profile_np_embeddings.append(np.array(emb_list))
            except json.JSONDecodeError:
                print(f"[Debug] Warning: Could not parse user embedding string at index {i}")
            except TypeError:
                print(f"[Debug] Warning: User embedding is not a string at index {i} (Type: {type(emb_str)})")

        if user_profile_np_embeddings:
            average_user_embedding = np.mean(user_profile_np_embeddings, axis=0)
            print(f"[Debug] Average user embedding created successfully. (Dims: {average_user_embedding.shape})")
        else:
            print("[Debug] No valid user profile embeddings found after parsing.")
            average_user_embedding = None

    recommended_places = []
    
    # 여기서부터 중요: 왜 추천 리스트에 못 들어가는지 확인
    if average_user_embedding is not None:
        valid_embedding_count = 0
        for place in nearby_places:
            if place.get('embedding') is not None:
                try:
                    place_emb_list = json.loads(place['embedding'])
                    place_np_embedding = np.array(place_emb_list)

                    user_emb_2d = average_user_embedding.reshape(1, -1)
                    place_emb_2d = place_np_embedding.reshape(1, -1)

                    similarity = cosine_similarity(user_emb_2d, place_emb_2d)[0][0]
                    valid_embedding_count += 1

                    recommended_places.append({
                        'place_id': place['place_id'],
                        'similarity_score': float(similarity),
                    })
                except json.JSONDecodeError:
                    print(f"[Debug] Place ID {place['place_id']}: JSON Decode Error")
                except Exception as e:
                     print(f"[Debug] Place ID {place['place_id']}: Error {e}")
            else:
                # 임베딩이 없어서 스킵되는 경우 로그 출력 (너무 많으면 주석 처리)
                # print(f"[Debug] Skipping Place ID {place['place_id']}: Embedding is None")
                pass
        
        print(f"[Debug] Total places with valid embeddings: {valid_embedding_count}")

        recommended_places.sort(key=lambda x: x['similarity_score'], reverse=True)

    else:
        print("[Debug] Cannot calculate recommendations: Average user embedding is None.")
    
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
    return itinerary_place_ids

# Function to exclude existing itinerary places
def exclude_existing_itinerary_places(recommended_places, itinerary_place_ids):
    filtered_recommendations = [
        place for place in recommended_places 
        if place['place_id'] not in itinerary_place_ids
    ]
    return filtered_recommendations


# ---------------------------------------------------------
# Main Recommendation Function
# ---------------------------------------------------------
def recommend_next_place(input_data):
    
    # [수정됨] 1. determine_anchor_location 호출하여 기준 좌표 재설정
    # 이 부분이 추가된 로직입니다.
    anchor_lat, anchor_lng = determine_anchor_location(
        input_data['itinerary_id'],
        input_data['day_id'],
        input_data['lat'],
        input_data['lng'],
        supabase
    )
    
    # 계산된 기준 좌표로 input_data 갱신
    input_data['lat'] = anchor_lat
    input_data['lng'] = anchor_lng
    
    # 2. Filter places by proximity (갱신된 좌표 기준 검색)
    nearby_places = get_nearby_places(input_data, all_places, haversine_distance)
    if not nearby_places:
        return "1.5km내 장소가 없습니다"

    # 3. Generate user profile from saved/liked places
    user_profile_embeddings = get_user_profile_embeddings(input_data['user_id'], all_places, supabase)
    if not user_profile_embeddings:
        return "저장하거나 좋아요한 장소가 없습니다"

    # 4. Calculate recommendation scores
    recommended_places = calculate_recommendation_scores(user_profile_embeddings, nearby_places)
    if not recommended_places:
        return "추천할 장소를 찾지 못했습니다."

    # 5. Exclude existing itinerary places
    itinerary_place_ids = get_itinerary_places(input_data['itinerary_id'], supabase)
    final_recommendations = exclude_existing_itinerary_places(recommended_places, itinerary_place_ids)
    
    if not final_recommendations:
        return "추천할 장소가 모두 여정에 등록된 장소입니다."
    
    return final_recommendations

if __name__ == "__main__":
    # Example input data
    input_data = {
        "user_id" : "7efc9d982d34d46648ece81e972ebeddafc8c0b061708a6e5a0468c29419452c",
        "lat" : 35.149548,
        "lng" : 126.922151,
        "itinerary_id" : 131,
        "day_id" : 148
    }

    print("--- Testing recommend_next_place ---")
    final_recommendations = recommend_next_place(input_data)

    print("\n--- Final Recommendations --- ")
    if isinstance(final_recommendations, str):
        print(final_recommendations)
    else:
        for i, rec_place in enumerate(final_recommendations):
            if i < 10: 
                print(f"Place ID: {rec_place['place_id']}, Score: {rec_place['similarity_score']:.4f}")