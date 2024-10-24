import pandas as pd
import googlemaps
from tqdm import tqdm
import json
import time

# Google API 키 설정
with open('../secrets.json') as f:
   secrets = json.loads(f.read())
GOOGLE_API_KEY = secrets['GOOGLE_API_KEY']

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# 데이터 로드
df = pd.read_csv('review_data_cleaned.csv', encoding='cp949')

# 처음 10개 가게만 선택
test_df = df.head(10)

def get_place_reviews(row):
   try:
       # 검색어 최적화 (가게이름 + 제주)
       search_query = f"{row['MCT_NM']} 제주도"
       print(f"\n검색 시도: {search_query}")
       
       # 장소 검색
       place_result = gmaps.places(search_query)
       
       if not place_result['results']:
           print(f"장소를 찾을 수 없음: {row['MCT_NM']}")
           return []
       
       place_id = place_result['results'][0]['place_id']
       print(f"장소 ID 찾음: {place_id}")
       
       # 상세 정보 가져오기
       place_details = gmaps.place(place_id)
       reviews = place_details.get('result', {}).get('reviews', [])
       
       # 리뷰 텍스트 추출
       review_texts = [review.get('text', '') for review in reviews if review.get('text')]
       print(f"리뷰 {len(review_texts)}개 수집")
       
       return review_texts
       
   except Exception as e:
       print(f"에러 발생 ({row['MCT_NM']}): {str(e)}")
       return []

# 결과 저장용 딕셔너리
results = {}

# 각 가게별로 리뷰 수집
for idx, row in test_df.iterrows():
   print(f"\n처리 중: {row['MCT_NM']}")
   reviews = get_place_reviews(row)
   results[row['MCT_NM']] = reviews
   
   # API 호출 제한 고려한 딜레이
   time.sleep(2)

# 결과 출력
print("\n=== 수집 결과 ===")
for place, reviews in results.items():
   print(f"\n{place}: {len(reviews)}개 리뷰")
   if reviews:
       print("첫 번째 리뷰:", reviews[0][:100] + "..." if len(reviews[0]) > 100 else reviews[0])

# 성공/실패 통계
success = sum(1 for reviews in results.values() if reviews)
print(f"\n총 {len(results)}개 중 {success}개 성공 ({success/len(results)*100:.1f}%)")