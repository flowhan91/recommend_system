#다른 사용자들은 여기도 방문했어요. 

import os
import json
import numpy as np
from typing import List, Dict, Optional
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class SimilarPlaceRecommender:
    def __init__(self):
        # 환경 변수에서 키를 가져오거나, 없으면 직접 입력 (보안상 환경변수 권장)
        self.url = "https://gvtmqaakgyorevninbmt.supabase.co"
        self.key = "sb_secret_WziBehEDoaGk4FOUGIYd5Q_qVJcs3Lj"
        
        if "YOUR_SUPABASE" in self.url:
             print(" .env 파일이 없거나 키가 설정되지 않았습니다.")
        
        self.supabase: Client = create_client(self.url, self.key)
        
        # 전체 장소 데이터 로드 (캐싱)
        self.all_places = self._fetch_all_places()

    def _fetch_all_places(self) -> List[Dict]:
        """모든 장소의 ID, 이름, 임베딩, 테마를 가져옵니다."""
        try:
            # 필요한 컬럼만 선택
            response = self.supabase.table("Place")\
                .select("place_id, name_kr, embedding, recommended_for")\
                .execute()
            
            # 임베딩이 있는 장소만 필터링
            valid_places = [p for p in response.data if p['embedding'] is not None]
            return valid_places
        except Exception as e:
            print(f"Error fetching places: {e}")
            return []

    def get_similar_places(self, target_place_id: int, top_k: int = 5) -> List[Dict]:
        """
        입력받은 place_id와 임베딩 유사도가 가장 높은 장소들을 반환합니다.
        """
        # 1. 타겟 장소 찾기
        target_place = next((p for p in self.all_places if p['place_id'] == target_place_id), None)
        
        if not target_place:
            print(f"Place ID {target_place_id} not found in database.")
            return []

        if not target_place['embedding']:
            print(f"Place ID {target_place_id} has no vector embedding.")
            return []

        # 2. 타겟 벡터 변환 (1, Dimension) 형태
        target_vector = np.array(json.loads(target_place['embedding'])).reshape(1, -1)
        
        candidates = []

        # 3. 전체 장소와 유사도 비교
        for place in self.all_places:
            # 자기 자신은 추천 제외
            if place['place_id'] == target_place_id:
                continue
                
            try:
                candidate_vector = np.array(json.loads(place['embedding'])).reshape(1, -1)
                
                # 코사인 유사도 계산 (-1 ~ 1)
                similarity = cosine_similarity(target_vector, candidate_vector)[0][0]
                
                candidates.append({
                    'place_id': place['place_id'],
                    'place_name_kr': place.get('name_kr', 'Unknown'),
                    'themes': place.get('recommended_for', []),
                    'similarity_score': float(similarity)
                })
            except Exception:
                continue

        # 4. 유사도 순으로 정렬 (내림차순)
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        # 5. 상위 k개 반환
        return candidates[:top_k]

# --- 실행부 ---
if __name__ == "__main__":
    recommender = SimilarPlaceRecommender()
    
    # 테스트할 장소 ID 입력 (예: 특정 카페나 명소의 ID)
    INPUT_PLACE_ID = 323  # 테스트용 ID, 실제 DB에 있는 ID로 변경 필요

    print(f"\n🔍 Searching for places similar to Place ID: {INPUT_PLACE_ID}...")
    
    results = recommender.get_similar_places(INPUT_PLACE_ID, top_k=5)
    
    if results:
        print(f"\n Top 5 Similar Places:")
        print("-" * 60)
        for idx, item in enumerate(results, 1):
            print(f"{idx}. [{item['place_name_kr']}] (ID: {item['place_id']})")
            print(f"   Score: {item['similarity_score']:.4f}")
            print(f"   Themes: {item['themes']}")
            print("-" * 60)
    else:
        print("\n No recommendations found.")