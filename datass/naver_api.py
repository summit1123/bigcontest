import pandas as pd
import urllib.request
import urllib.parse
import json
import time
import ssl
import requests
from typing import List, Dict

class NaverBlogReviewCollector:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/blog"
        self.headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        # SSL 컨텍스트 생성
        self.context = ssl._create_unverified_context()

    def search_blog(self, query: str, display: int = 5) -> Dict:
        """네이버 블로그 검색 API를 호출하여 결과를 반환합니다."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"{self.base_url}?query={encoded_query}&display={display}&sort=sim"
            
            # requests 라이브러리 사용
            response = requests.get(url, headers=self.headers, verify=False)
            response.raise_for_status()  # HTTP 에러 체크
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패 ({query}): {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패 ({query}): {str(e)}")
            return None
        except Exception as e:
            print(f"예상치 못한 에러 ({query}): {str(e)}")
            return None

    def extract_reviews(self, search_result: Dict) -> List[str]:
        """검색 결과에서 리뷰 텍스트 뽑기"""
        if not search_result or 'items' not in search_result:
            return []
        
        reviews = []
        for item in search_result['items']:
            # HTML 태그 제거 및 텍스트 정제
            desc = item['description'].replace('<b>', '').replace('</b>', '')
            # 중복 공백 제거 및 앞뒤 공백 제거
            desc = ' '.join(desc.split()).strip()
            if desc:  # 빈 문자열이 아닌 경우만 추가
                reviews.append(desc)
        
        return reviews

    def collect_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터에서 리뷰 없는거 체크"""
        updated_df = df.copy()
        
        # 리뷰가 없는 행 필터링 (문자열 '[]'도 체크)
        mask = updated_df['REVIEW'].apply(lambda x: x == [] or x == '[]' or pd.isna(x))
        restaurants_without_reviews = updated_df[mask]
        
        print(f"리뷰 수집이 필요한 식당 수: {len(restaurants_without_reviews)}")
        
        # 진행상황 표시를 위한 카운터
        total = len(restaurants_without_reviews)
        success_count = 0
        
        for idx, row in restaurants_without_reviews.iterrows():
            try:
                # 검색어 생성 (식당이름 + 주소 일부)
                addr_parts = row['ADDR'].split()
                if len(addr_parts) >= 2:
                    search_query = f"{row['NAME']} {addr_parts[0]} {addr_parts[1]} 후기"
                else:
                    search_query = f"{row['NAME']} {row['ADDR']} 후기"
                
                # API 호출
                search_result = self.search_blog(search_query)
                
                if search_result:
                    reviews = self.extract_reviews(search_result)
                    if reviews:
                        updated_df.at[idx, 'REVIEW'] = reviews
                        success_count += 1
                        print(f"[{success_count}/{total}] 리뷰 수집 완료: {row['NAME']}")
                    else:
                        print(f"[{success_count}/{total}] 리뷰 없음: {row['NAME']}")
                
                # 중간 저장 (10개마다)
                if success_count % 10 == 0:
                    updated_df.to_csv('data/review_data_updated_temp.csv', 
                                    index=False, 
                                    encoding='utf-8-sig')
                
                # API 호출 제한 고려
                time.sleep(0.1)
            
            except Exception as e:
                print(f"Error processing {row['NAME']}: {str(e)}")
                continue
        
        return updated_df

def try_read_csv(file_path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'cp949', 'euc-kr']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            continue
    
    raise ValueError(f"Could not read file with any of the encodings: {encodings}")

def main():
    try:
        # requests의 경고 메시지 무시
        import warnings
        warnings.filterwarnings('ignore')
        
        # SSL 경고 무시
        requests.packages.urllib3.disable_warnings()
        
        print("CSV 파일 로딩 중...")
        df = try_read_csv('data/review_data_cleaned.csv')
        print("CSV 파일 로딩 완료")
        
        print(f"\n데이터 크기: {df.shape}")
        print("컬럼 목록:", df.columns.tolist())
        
        # 네이버 API 인증정보
        CLIENT_ID = "soIPvzuKxuZLfLV1bFLq"
        CLIENT_SECRET = "etvERcpaAr"
        
        collector = NaverBlogReviewCollector(CLIENT_ID, CLIENT_SECRET)
        updated_df = collector.collect_reviews(df)
        
        # 최종 결과 저장
        updated_df.to_csv('data/review_data_updated.csv', 
                         index=False, 
                         encoding='utf-8-sig')
        
        # 수집 결과 출력
        total_reviews = len(updated_df[updated_df['REVIEW'].apply(
            lambda x: x != [] and x != '[]' and not pd.isna(x))])
        print(f"\n전체 식당 수: {len(updated_df)}")
        print(f"리뷰 보유 식당 수: {total_reviews}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()