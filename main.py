from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import recommender_location
import recommender_intinerary
import supabase

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

#=====주문서 양식=====
#테스트용
class PlaceDetailsRequest(BaseModel):
    liked_places: List[int]  # 예: [1, 5, 10]

#위치기반 추천
class LocationRequestModel(BaseModel):
    request: str
    user_id: str
    lat: float  # 위도는 실수형
    lng: float  # 경도는 실수형
#여정 장소추천
# Pydantic model for itinerary recommendation request
class ItineraryRecommendationRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    itinerary_id: int
    day_id: int


@app.get("/")
def health_check():
    return {"status": "Active"}

#위치기반 추천
@app.post("/recommend/location",response_model=List[Dict[str, Any]])
async def get_location_based_recommendations(input_data: LocationRequestModel):
    """
    위치 정보를 받아 추천 장소를 반환하는 API
    """
    try:
        # Pydantic 모델을 딕셔너리 형태로 변환 (함수 인풋 형식에 맞춤)
        location_input = input_data.dict()
        
        # 요청 타입 확인 (선택 사항)
        if location_input.get("request") != "location":
            raise HTTPException(status_code=400, detail="Invalid request type")

        # recommender_location 모듈의 함수 호출
        recommendation_results = recommender_location.get_recommendations(location_input)
        
        return recommendation_results

    except Exception as e:
        # 에러 발생 시 처리
        raise HTTPException(status_code=500, detail=str(e))

# Itinerary Recommendation
@app.post("/recommend/itinerary", response_model=List[Dict[str, Any]])
async def recommend_next_itinerary_place(request_data: ItineraryRecommendationRequest):
    """
    User-specific itinerary recommendation based on location, saved/liked places,
    and excluding existing itinerary places.
    """
    try:
        # Convert Pydantic model to dict for the recommender function
        input_data = request_data.dict()

        # Call the recommendation logic from recommender_itinerary
        # The recommender_itinerary module initializes its own supabase client and all_places
        recommendations = recommender_intinerary.recommend_next_place(input_data)

        if isinstance(recommendations, str):
            # Handle cases where the recommender returns an error string
            raise HTTPException(status_code=404, detail=recommendations)
        else:
            # Ensure similarity_score is a standard float for JSON serialization
            for rec_place in recommendations:
                if 'similarity_score' in rec_place:
                    rec_place['similarity_score'] = float(rec_place['similarity_score'])
            return recommendations
    except Exception as e:
        print(f"Error during itinerary recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")