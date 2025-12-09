import math
import json
import os
import numpy as np
from typing import List, Dict, Optional
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PLACE_THEME_COLUMN = "recommended_for" 

class LocationRecommendationEngine:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")
        self.supabase: Client = create_client(self.url, self.key)
        
        # Load places
        self.all_places = self._fetch_all_places()

    def _fetch_all_places(self) -> List[Dict]:
        try:
            # [수정됨] 컬럼명을 'name_kr'로 변경
            query = f"place_id, lat, lng, embedding, name_kr, {PLACE_THEME_COLUMN}"
            
            response = self.supabase.table("Place")\
                .select(query)\
                .execute()
            
            return [p for p in response.data if p['lat'] is not None and p['lng'] is not None]
        except Exception as e:
            print(f"Error fetching places: {e}")
            return []

    def _haversine_vectorized(self, user_lat, user_lng, places_data, radius_km=1.5):
        if not places_data:
            return []

        R = 6371
        u_lat_rad = math.radians(user_lat)
        u_lng_rad = math.radians(user_lng)
        
        lats = np.array([p['lat'] for p in places_data], dtype=float)
        lngs = np.array([p['lng'] for p in places_data], dtype=float)
        lats_rad = np.radians(lats)
        lngs_rad = np.radians(lngs)
        
        dlat = lats_rad - u_lat_rad
        dlng = lngs_rad - u_lng_rad
        
        a = np.sin(dlat/2)**2 + np.cos(u_lat_rad) * np.cos(lats_rad) * np.sin(dlng/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = R * c
        
        nearby_indices = np.where(distances <= radius_km)[0]
        return [places_data[i] for i in nearby_indices]

    def get_user_themes(self, user_id: str) -> List[str]:
        try:
            response = self.supabase.table("Users")\
                .select("themes").eq("user_id", user_id).single().execute()
            if response.data and response.data.get('themes'):
                return [str(t).lower() for t in response.data['themes']]
            return []
        except Exception:
            return []

    def _is_theme_match(self, user_themes: List[str], place_info: Dict) -> bool:
        place_tags = place_info.get(PLACE_THEME_COLUMN)
        
        if not place_tags:
            return False

        if isinstance(place_tags, list):
            place_tags_str = " ".join([str(t).lower() for t in place_tags])
        else:
            place_tags_str = str(place_tags).lower()

        for u_theme in user_themes:
            if u_theme in place_tags_str:
                print("theme matches place")
                return True
        return False

    def get_user_profile_vector(self, user_id: str, user_themes: List[str]) -> Optional[np.ndarray]:
        try:
            saved = self.supabase.table("SavedPlaces").select("place_id").eq("user_id", user_id).execute()
            liked = self.supabase.table("LikedPlaces").select("place_id").eq("user_id", user_id).execute()
            
            interaction_ids = {item['place_id'] for item in saved.data + liked.data}
            
            if not interaction_ids:
                return None
            
            embeddings = []
            weights = []

            for p in self.all_places:
                if p['place_id'] in interaction_ids and p['embedding']:
                    try:
                        emb = np.array(json.loads(p['embedding']))
                        
                        weight = 1.0
                        if self._is_theme_match(user_themes, p):
                            weight = 3.0 
                        
                        embeddings.append(emb)
                        weights.append(weight)
                    except:
                        continue
            
            if not embeddings:
                return None
            
            return np.average(embeddings, axis=0, weights=weights)

        except Exception:
            return None

    def recommend(self, input_data: Dict) -> List[Dict]:
        user_lat = input_data['lat']
        user_lng = input_data['lng']
        user_id = input_data['user_id']
        
        nearby_places = self._haversine_vectorized(user_lat, user_lng, self.all_places, radius_km=1.5)
        if not nearby_places:
            return []

        user_themes = self.get_user_themes(user_id)
        user_vector = self.get_user_profile_vector(user_id, user_themes)
        
        if user_vector is None:
            return []

        recommendations = []
        user_vector_2d = user_vector.reshape(1, -1)
        
        for place in nearby_places:
            if not place.get('embedding'):
                continue
            try:
                place_vec = np.array(json.loads(place['embedding'])).reshape(1, -1)
                sim = cosine_similarity(user_vector_2d, place_vec)[0][0]
                
                # [수정됨] DB 컬럼 'name_kr'을 읽어서 출력 키 'place_name_kr'에 매핑
                recommendations.append({
                    'place_id': place['place_id'],
                    'place_name_kr': place.get('name_kr', 'Unknown'), 
                    'similarity_score': float(sim)
                })
            except:
                continue
        
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations

if __name__ == "__main__":
    engine = LocationRecommendationEngine()
    
    input_data = {
        "user_id" : "7efc9d982d34d46648ece81e972ebeddafc8c0b061708a6e5a0468c29419452c",
        "lat" : 35.149548,
        "lng" : 126.922151,
    }

    results = engine.recommend(input_data)
    
    print("\n[Recommendation Results]")
    for item in results:
        # 출력 시 한글 이름 확인
        print(f"ID: {item['place_id']} | Name: {item['place_name_kr']} | Score: {item['similarity_score']:.4f}")