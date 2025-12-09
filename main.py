from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import recommender_location
import recommender_intinerary
import supabase
import recommender_place

load_dotenv()

app = FastAPI()
recommendation_engine = recommender_intinerary.RecommendationEngine()

SUPABASE_URL = "https://gvtmqaakgyorevninbmt.supabase.co"
SUPABASE_KEY = "sb_secret_WziBehEDoaGk4FOUGIYd5Q_qVJcs3Lj"
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

#유사장소 추천
class PlaceRecommendationRequest(BaseModel):
    place_id:int


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

@app.post("/recommend/itinerary", response_model=List[Dict[str, Any]])
async def recommend_next_itinerary_place(request_data: ItineraryRecommendationRequest):
    """
    User-specific itinerary recommendation using the RecommendationEngine class.
    """
    try:
        # Pydantic 모델을 딕셔너리로 변환 (엔진 입력 형식에 맞춤)
        input_data = request_data.dict()

        # 2. 전역 변수로 만들어둔 엔진 인스턴스를 사용하여 추천 로직 실행
        # 기존: recommender_intinerary.recommend(input_data)
        # 변경: recommendation_engine.recommend(input_data)
        recommendations = recommendation_engine.recommend(input_data)

        # 결과 처리 로직 (기존 코드 유지)
        if isinstance(recommendations, str):
            # 에러 메시지가 문자열로 반환된 경우 처리
            raise HTTPException(status_code=404, detail=recommendations)
        else:
            # JSON 직렬화를 위해 float 변환 처리
            for rec_place in recommendations:
                if 'similarity_score' in rec_place:
                    rec_place['similarity_score'] = float(rec_place['similarity_score'])
            return recommendations

    except Exception as e:
        print(f"Error during itinerary recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/recommend/place", response_model=List[Dict[str, Any]])
async def recommend_similar_places(request_data:PlaceRecommendationRequest):
    input_data = request_data.dict()
    recommendations = recommender_place.get_similar_places(input_data['place_id'], top_k=10)