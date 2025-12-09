from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv

# 사용자 정의 추천 모듈 임포트
import recommender_location
import recommender_intinerary
import recommender_place

# 1. 환경변수 로드
load_dotenv()

app = FastAPI()

# -----------------------------------------------------------
# [Global Init] 서버 시작 시 엔진 인스턴스 생성 (Singleton 패턴)
# 각 클래스의 __init__이 실행되면서 DB 데이터를 메모리에 적재합니다.
# -----------------------------------------------------------
print("Initializing Recommendation Engines...")
itinerary_engine = recommender_intinerary.RecommendationEngine()
location_engine = recommender_location.LocationRecommendationEngine()
place_engine = recommender_place.SimilarPlaceRecommender() # 혹은 PlaceRecommendationEngine()
print("Engines Ready.")


# -----------------------------------------------------------
# [Pydantic Models] 요청/응답 스키마 정의
# -----------------------------------------------------------

# 위치기반 추천 요청
class LocationRequestModel(BaseModel):
    request: str
    user_id: str
    lat: float
    lng: float

# 여정 추천 요청
class ItineraryRecommendationRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    itinerary_id: int
    day_id: int

# 유사장소 추천 요청
class PlaceRecommendationRequest(BaseModel):
    place_id: int


# -----------------------------------------------------------
# [API Endpoints]
# -----------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "Active"}

# 1. 위치기반 추천 API
@app.post("/recommend/location", response_model=List[Dict[str, Any]])
async def get_location_based_recommendations(input_data: LocationRequestModel):
    """
    위치 정보를 받아 추천 장소를 반환하는 API
    """
    try:
        location_input = input_data.dict()
        
        if location_input.get("request") != "location":
            raise HTTPException(status_code=400, detail="Invalid request type")

        # 전역 location_engine 사용
        recommendation_results = location_engine.recommend(location_input)
        return recommendation_results

    except Exception as e:
        print(f"Error processing location recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 2. 여정 추천 API (수정된 recommender_itinerary.py 연동)
@app.post("/recommend/itinerary", response_model=List[Dict[str, Any]])
async def recommend_next_itinerary_place(request_data: ItineraryRecommendationRequest):
    """
    User-specific itinerary recommendation using the RecommendationEngine class.
    """
    try:
        input_data = request_data.dict()

        # 전역 itinerary_engine 사용
        recommendations = itinerary_engine.recommend(input_data)

        # 오류 처리 (문자열이 반환되면 에러로 간주)
        if isinstance(recommendations, str):
            raise HTTPException(status_code=404, detail=recommendations)
        
        # recommender_itinerary.py에서 이미 float 변환을 했으므로 바로 반환 가능
        return recommendations

    except Exception as e:
        print(f"Error during itinerary recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# 3. 유사 장소 추천 API
@app.post("/recommend/place", response_model=List[Dict[str, Any]])
async def recommend_similar_places(request_data: PlaceRecommendationRequest):
    try:
        input_data = request_data.dict()
        
        # 전역 place_engine 사용
        recommendations = place_engine.get_similar_places(input_data['place_id'], top_k=10)
        
        # [중요 수정] 결과를 return 해야 합니다.
        return recommendations

    except Exception as e:
        print(f"Error during place recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))