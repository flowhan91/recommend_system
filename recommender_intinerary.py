import math
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

PLACE_THEME_COLUMN = "recommended_for" 

class RecommendationEngine:
    def __init__(self):
        self.url = "deleted"
        self.key = "deleted"
        self.supabase: Client = create_client(self.url, self.key)
        self.all_places = self._fetch_all_places()

    def _fetch_all_places(self) -> List[Dict]:
        try:
            query = f"place_id, lat, lng, embedding, name_kr, {PLACE_THEME_COLUMN}"
            response = self.supabase.table("Place").select(query).execute()
            return [p for p in response.data if p['lat'] is not None and p['lng'] is not None]
        except Exception as e:
            print(f"Error fetching places: {e}")
            return []

    def _haversine_vectorized(self, user_lat, user_lng, places_data, radius_km=1.5):
        if not places_data: return []
        R = 6371
        u_lat_rad = math.radians(user_lat)
        u_lng_rad = math.radians(user_lng)
        lats = np.array([p['lat'] for p in places_data], dtype=float)
        lngs = np.array([p['lng'] for p in places_data], dtype=float)
        dlat = np.radians(lats) - u_lat_rad
        dlng = np.radians(lngs) - u_lng_rad
        a = np.sin(dlat/2)**2 + np.cos(u_lat_rad) * np.cos(np.radians(lats)) * np.sin(dlng/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = R * c
        nearby_indices = np.where(distances <= radius_km)[0]
        return [places_data[i] for i in nearby_indices]

    def _normalize_tags(self, raw_data) -> Set[str]:
        if not raw_data: return set()
        normalized = set()
        if isinstance(raw_data, list):
            for item in raw_data: normalized.add(str(item).lower().strip())
        elif isinstance(raw_data, str):
            for part in raw_data.split(','): normalized.add(part.lower().strip())
        return normalized

    def get_user_themes(self, user_id: str) -> Set[str]:
        try:
            response = self.supabase.table("Users").select("themes").eq("user_id", user_id).single().execute()
            if response.data: return self._normalize_tags(response.data.get('themes'))
            return set()
        except: return set()

    def get_itinerary_theme(self, itinerary_id: int) -> Set[str]:
        try:
            response = self.supabase.table("Itinerary").select("theme").eq("itinerary_id", itinerary_id).single().execute()
            if response.data: return self._normalize_tags(response.data.get('theme'))
            return set()
        except: return set()

    def _is_theme_match(self, active_themes: Set[str], place_info: Dict) -> bool:
        # Used for Profile Boosting only
        place_tags_set = self._normalize_tags(place_info.get(PLACE_THEME_COLUMN))
        if not place_tags_set: return False
        place_tags_joined = " ".join(place_tags_set)
        for theme in active_themes:
            if theme in place_tags_set or theme in place_tags_joined: return True
        return False

    def get_user_profile_vector(self, user_id: str, active_themes: Set[str]) -> Optional[np.ndarray]:
        try:
            saved = self.supabase.table("SavedPlaces").select("place_id").eq("user_id", user_id).execute()
            liked = self.supabase.table("LikedPlaces").select("place_id").eq("user_id", user_id).execute()
            interaction_ids = {item['place_id'] for item in saved.data + liked.data}
            if not interaction_ids: return None
            
            embeddings, weights = [], []
            for p in self.all_places:
                if p['place_id'] in interaction_ids and p['embedding']:
                    try:
                        emb = np.array(json.loads(p['embedding']))
                        weight = 3.0 if self._is_theme_match(active_themes, p) else 1.0
                        embeddings.append(emb)
                        weights.append(weight)
                    except: continue
            
            if not embeddings: return None
            return np.average(embeddings, axis=0, weights=weights)
        except: return None

    def determine_anchor_location(self, itinerary_id, day_id, current_lat, current_lng):
        try:
            day_p = self.supabase.table("ItineraryPlace").select("place_id").eq("itinerary_id", itinerary_id).eq("day_id", day_id).execute()
            if day_p.data: return self._get_place_coords(day_p.data[-1]['place_id'])
            itin_p = self.supabase.table("ItineraryPlace").select("place_id").eq("itinerary_id", itinerary_id).execute()
            if itin_p.data: return self._get_place_coords(itin_p.data[-1]['place_id'])
        except: pass
        return current_lat, current_lng

    def _get_place_coords(self, place_id):
        res = self.supabase.table("Place").select("lat, lng").eq("place_id", place_id).single().execute()
        return (res.data['lat'], res.data['lng']) if res.data else (0.0, 0.0)

    def recommend(self, input_data: Dict) -> List[Dict]:
        anchor_lat, anchor_lng = self.determine_anchor_location(
            input_data['itinerary_id'], input_data['day_id'], input_data['lat'], input_data['lng']
        )
        
        nearby_places = self._haversine_vectorized(anchor_lat, anchor_lng, self.all_places, radius_km=1.5)
        if not nearby_places: return []

        user_themes = self.get_user_themes(input_data['user_id'])
        itinerary_themes = self.get_itinerary_theme(input_data['itinerary_id'])
        active_themes = user_themes.union(itinerary_themes)
        
        user_vector = self.get_user_profile_vector(input_data['user_id'], active_themes)
        if user_vector is None: return []

        recommendations = []
        user_vector_2d = user_vector.reshape(1, -1)
        
        for place in nearby_places:
            if not place.get('embedding'): continue
            try:
                place_vec = np.array(json.loads(place['embedding'])).reshape(1, -1)
                sim = cosine_similarity(user_vector_2d, place_vec)[0][0]

                # [NEW] Explicitly check which themes match for the CANDIDATE place
                matched_list = []
                place_tags_set = self._normalize_tags(place.get(PLACE_THEME_COLUMN))
                place_tags_joined = " ".join(place_tags_set)
                
                for theme in active_themes:
                    if theme in place_tags_set or theme in place_tags_joined:
                        matched_list.append(theme)

                recommendations.append({
                    'place_id': place['place_id'],
                    'place_name_kr': place.get('name_kr', 'Unknown'),
                    'similarity_score': float(sim),
                    'matched_themes': matched_list  # Result: ['shopping', 'food']
                })
            except: continue
        
        visited_res = self.supabase.table("ItineraryPlace").select("place_id").eq("itinerary_id", input_data['itinerary_id']).execute()
        visited_ids = {item['place_id'] for item in visited_res.data}
        
        final_recs = [r for r in recommendations if r['place_id'] not in visited_ids]
        final_recs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return final_recs

if __name__ == "__main__":
    engine = RecommendationEngine()
    input_data = {
        "user_id" : "7efc9d982d34d46648ece81e972ebeddafc8c0b061708a6e5a0468c29419452c",
        "lat" : 35.149548, "lng" : 126.922151, "itinerary_id" : 131, "day_id" : 148
    }

    results = engine.recommend(input_data)
    
    print("\n[Recommendation Results]")
    for item in results:
        # Now you can see matched themes in the print
        print(f"ID: {item['place_id']} | Name: {item['place_name_kr']} | Match: {item['matched_themes']} | Score: {item['similarity_score']:.4f}")
