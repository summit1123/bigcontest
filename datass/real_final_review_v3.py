import pandas as pd
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from datetime import datetime
import os
import time
import logging
import json
import tiktoken
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

class RestaurantAnalyzer:
    def __init__(self, api_key: str):
        """초기화"""
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.token_counter = tiktoken.encoding_for_model("gpt-4o-mini")
        self.setup_processing_params()

    def setup_processing_params(self):
        """처리 파라미터 설정"""
        self.batch_size = 5
        self.concurrent_requests = 3
        self.max_retries = 5
        self.retry_delay = 2
        self.progress_save_interval = 10
        
        # 키워드 카테고리 정의
        self.keyword_categories = {
            'TASTE': ['담백한', '얼큰한', '고소한', '달달한', '짭짤한', '깔끔한', '시원한', '개운한', '진한', '건강한', '감칠맛', '쫄깃한', '바삭한', '부드러운', '촉촉한', '아삭한', '푸짐한'],
            'JEJU_SPECIALTY': ['흑돼지', '갈치', '고등어', '한치', '전복', '옥돔', '한라봉', '감귤', '녹차', '당근', '제주식', '향토음식', '제주식재료', '로컬맛집', '제주전통'],
            'RESTAURANT_STYLE': ['모던한', '전통적인', '캐주얼한', '고급스러운', '아기자기한', '빈티지', '깔끔한', '자연친화적', '감성적인', '이색적인', '룸있는', '테라스', '바테이블', '좌식', '입식', '오션뷰'],
            'VISIT_ATTRIBUTES': ['데이트', '가족식사', '친구모임', '회식', '혼밥', '관광', '아침식사', '브런치', '점심특선', '저녁식사', '야간영업', '소규모', '단체가능', '가족단위', '프라이빗'],
            'CONVENIENCE': ['주차가능', '대기공간', '무선wifi', '콘센트', '유아시설', '예약필수', '예약가능', '예약불가', '선착순', '대중교통', '도보접근', '주차용이'],
            'TREND': ['인스타감성', '포토존', '뷰맛집', '핫플레이스', '숨은맛집'],
            'PRICE_VALUE': ['가성비좋은', '고급가격대', '중저가', '착한가격', '가격부담'],
            'ATMOSPHERE': ['조용한', '시끌벅적한', '프라이빗한', '널찍한', '아늑한', '오션뷰', '시티뷰', '가든뷰', '야경맛집'],
            'SERVICE': ['친절한', '전문적인', '설명잘하는', '빠른서비스', '정갈한준비', '맞춤서비스'],
            'SEASONAL': ['제철재료', '계절특선', '시즌한정', '봄여행', '여름휴가', '가을나들이', '겨울여행'],
            'TIME_OF_DAY': ['24시간', '심야영업', '올데이', '아침추천', '점심추천', '저녁추천', '야식추천']
        }

    def validate_row(self, row: pd.Series) -> None:
        """데이터 유효성 검증"""
        if 'MCT_NM' not in row or pd.isna(row['MCT_NM']):
            raise ValueError("Restaurant name (MCT_NM) is missing or empty")
        if 'REVIEW' not in row:
            row['REVIEW'] = ''
        if 'DESC' not in row:
            row['DESC'] = ''

    def create_prompt(self, row: pd.Series) -> str:
        """프롬프트 생성"""
        self.validate_row(row)
        prompt = f"""식당: {row['MCT_NM']}
리뷰: {row['REVIEW']}
설명: {row['DESC'] if pd.notna(row['DESC']) else ''}

[분석 요청사항]
1. 리뷰 요약 (2-3문장):
- 방문자 평가/만족도
- 대표메뉴 맛/품질
- 가격 대비 가치
- 매장 분위기/서비스

2. 설명 요약 (2-3문장):
- 대표 특징/주력메뉴
- 차별화 포인트

3. 키워드 분석:
주어진 리뷰와 설명을 바탕으로 각 카테고리별로 명확한 근거가 있는 키워드만 선택하세요.
실제 언급되거나 명확히 유추할 수 있는 내용만 선택하고, 해당 내용이 없다면 빈 값으로 두세요.
각 카테고리당 최대 3개까지만 선택 가능합니다.

[키워드 카테고리]"""

        for category, keywords in self.keyword_categories.items():
            prompt += f"\n{category}: [{', '.join(keywords)}]"

        prompt += """

[답변 형식]
REVIEW_SUMMARY: 요약문
DESC_SUMMARY: 설명문
KEYWORDS:"""

        for category in self.keyword_categories.keys():
            prompt += f"\n{category}: [키워드들 또는 빈값]"

        prompt += """

[키워드 선택 기준]
1. 리뷰/설명에서 직접적으로 언급된 내용만 선택하시오. 
2. 맥락상 명확하게 유추 가능한 내용만 추가하시오.
3. 객관적 근거가 있는 내용만 선택하시오.
4. 불확실하거나 모호한 경우 선택하지 않도록 하시오.
5. 각 카테고리별로 관련 내용이 없다면 반드시 빈 값으로 둘 것"""

        return prompt

    def save_progress(self, df: pd.DataFrame, current_count: int, total_rows: int):
        """중간 진행상황 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_dir = Path('data/progress')
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_file = progress_dir / f'review_data_progress_{timestamp}.csv'
            df.to_csv(progress_file, index=False, encoding='utf-8-sig')
            print(f"\n중간 결과 저장 완료 ({current_count}/{total_rows}): {progress_file}")
        except Exception as e:
            logging.error(f"진행상황 저장 중 오류 발생: {str(e)}")
            print(f"\n진행상황 저장 실패: {str(e)}")

    def print_sample_results(self, results: List[Dict], original_data: pd.DataFrame):
        """샘플 결과 출력"""
        print("\n=== 최근 처리된 결과 샘플 ===")
        for idx, result in enumerate(results):
            if not result.get('error'):
                restaurant_name = original_data.iloc[idx]['MCT_NM']
                print(f"\n식당명: {restaurant_name}")
                print(f"리뷰 요약: {result['review_summary'][:100]}...")
                print(f"설명 요약: {result['desc_summary'][:100]}...")
                print(f"추출 키워드: {result['keywords']}")
                print("-" * 50)

    async def process_restaurants(self, input_file: str = 'data/review_data_updated.csv'):
        """레스토랑 데이터 처리"""
        try:
            df = pd.read_csv(input_file)
            total_rows = len(df)
            
            print("\n" + "="*50)
            print(f"분석 시작")
            print(f"총 처리할 레스토랑: {total_rows}개")
            print("="*50 + "\n")
            
            processed = 0
            errors = 0
            start_time = time.time()
            
            df['review_summary'] = ''
            df['desc_summary'] = ''
            df['keywords'] = ''
            
            for i in range(0, total_rows, self.batch_size):
                chunk = df.iloc[i:min(i + self.batch_size, total_rows)]
                print(f"\n현재 처리중: {chunk['MCT_NM'].tolist()}")
                
                results = await self._process_batch(chunk)
                
                for idx, result in enumerate(results):
                    if result.get('error'):
                        errors += 1
                        continue
                        
                    row_idx = chunk.index[idx]
                    df.at[row_idx, 'review_summary'] = result['review_summary']
                    df.at[row_idx, 'desc_summary'] = result['desc_summary']
                    df.at[row_idx, 'keywords'] = result['keywords']
                    processed += 1
                
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / processed if processed > 0 else 0
                estimated_remaining = avg_time_per_item * (total_rows - processed)
                
                print(f"\n진행상황:")
                print(f"처리완료: {processed}/{total_rows} ({processed/total_rows*100:.1f}%)")
                print(f"에러: {errors}개")
                print(f"평균 처리시간: {avg_time_per_item:.1f}초/건")
                print(f"예상 남은시간: {estimated_remaining/60:.1f}분")
                
                self.print_sample_results(results, chunk)
                
                if processed % self.progress_save_interval == 0:
                    self.save_progress(df, processed, total_rows)
                
                await asyncio.sleep(1)
            
            output_file = input_file.replace('.csv', '_analyzed.csv')
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print("\n" + "="*50)
            print(f"분석 완료!")
            print(f"- 처리된 레스토랑: {processed}")
            print(f"- 에러 발생: {errors}")
            print(f"- 총 소요시간: {(time.time() - start_time)/60:.1f}분")
            print(f"- 결과 파일: {output_file}")
            print("="*50 + "\n")
            
            return df
            
        except Exception as e:
            logging.error(f"처리 중 오류 발생: {str(e)}")
            raise

    async def _process_batch(self, batch: pd.DataFrame) -> List[Dict]:
        """배치 처리"""
        async def process_row(row):
            try:
                content = self.create_prompt(row)
                for attempt in range(self.max_retries):
                    try:
                        response = await self.async_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a restaurant analysis expert. Respond in Korean."},
                                {"role": "user", "content": content}
                            ],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        text = response.choices[0].message.content.strip()
                        return self.parse_response(text)
                    except Exception as e:
                        if "429" in str(e):
                            wait_time = self.retry_delay * (2 ** attempt)
                            print(f"\nRate limit 발생. {wait_time}초 대기 후 재시도...")
                            await asyncio.sleep(wait_time)
                        elif attempt == self.max_retries - 1:
                            raise
                        else:
                            await asyncio.sleep(self.retry_delay)
            except Exception as e:
                error_msg = str(e)
                print(f"\n처리 중 에러 발생: {error_msg[:100]}...")
                return {
                    'review_summary': '',
                    'desc_summary': '',
                    'keywords': json.dumps({category: [] for category in self.keyword_categories.keys()}, ensure_ascii=False),
                    'error': error_msg
                }

        tasks = [process_row(row) for _, row in batch.iterrows()]
        return await asyncio.gather(*tasks)

    def parse_response(self, text: str) -> Dict:
        """응답 파싱"""
        try:
            parts = {}
            current_key = None
            current_value = []
            keywords = {category: [] for category in self.keyword_categories.keys()}
            
            lines = text.split('\n')
            in_keywords_section = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line == 'KEYWORDS:':
                    in_keywords_section = True
                    if current_key and current_value:
                        parts[current_key] = ' '.join(current_value)
                    continue
                    
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    if in_keywords_section and key in self.keyword_categories:
                        clean_value = value.strip('[]').strip()
                        if clean_value:
                            extracted_keywords = [k.strip() for k in clean_value.split(',') if k.strip()]
                            valid_keywords = [k for k in extracted_keywords if k in self.keyword_categories[key]]
                            keywords[key] = valid_keywords[:3]
                    elif key in ['REVIEW_SUMMARY', 'DESC_SUMMARY']:
                        if current_key and current_value:
                            parts[current_key] = ' '.join(current_value)
                        current_key = key
                        current_value = [value]
                else:
                    if current_key and not in_keywords_section:
                        current_value.append(line)
                        
            if current_key and current_value and not in_keywords_section:
                parts[current_key] = ' '.join(current_value)
                
            return {
                'review_summary': parts.get('REVIEW_SUMMARY', ''),
                'desc_summary': parts.get('DESC_SUMMARY', ''),
                'keywords': json.dumps(keywords, ensure_ascii=False),
                'error': None
            }
            
        except Exception as e:
            logging.error(f"Response parsing error: {str(e)}")
            return {
                'review_summary': '',
                'desc_summary': '',
                'keywords': json.dumps({category: [] for category in self.keyword_categories.keys()}, ensure_ascii=False),
                'error': f"Parsing error: {str(e)}"
            }

async def main():
    """메인 실행 함수"""
    try:
        with open('secrets.json') as f:
            config = json.load(f)
        analyzer = RestaurantAnalyzer(api_key=config['OPENAI_API_KEY'])
        await analyzer.process_restaurants()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())