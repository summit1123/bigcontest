import pandas as pd
import asyncio
from typing import List, Dict
from openai import OpenAI, AsyncOpenAI
from datetime import datetime
import os
from tqdm import tqdm
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
        self.token_counter = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
        self.setup_processing_params()
        
    def setup_processing_params(self):
        """처리 파라미터 설정"""
        self.batch_size = 5
        self.concurrent_requests = 3
        self.max_retries = 5
        self.retry_delay = 2
        self.progress_save_interval = 10  # 10개 처리마다 중간 결과 저장
        
    def create_prompt(self, row: pd.Series) -> str:
        """최적화된 프롬프트 생성"""
        return f"""
식당 이름: {row['MCT_NM']}

리뷰:
{row['REVIEW']}

설명:
{row['DESC'] if pd.notna(row['DESC']) else ''}

다음 세 가지 분석을 간단히 해주세요:

1. 리뷰 요약 (2-3문장):
2. 설명 요약 (1문장):
3. 키워드 (아래 카테고리에서만 선택, 각 최대 2개):
- VIEW: [ocean_view, sunset_view, mountain_view]
- INTERIOR: [modern, vintage, traditional, luxury, casual, romantic]
- TASTE: [spicy, rich, light, sweet, savory]
- SPECIALTY: [jeju_ingredients, seasonal, handmade]

형식:
REVIEW_SUMMARY: [리뷰 요약]
DESC_SUMMARY: [설명 요약]
KEYWORDS: 
VIEW: keyword1[, keyword2]
INTERIOR: keyword1[, keyword2]
TASTE: keyword1[, keyword2]
SPECIALTY: keyword1[, keyword2]"""

    def save_progress(self, df: pd.DataFrame, current_count: int, total_rows: int):
        """중간 진행상황 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_file = f'data/progress/review_data_progress_{timestamp}.csv'
        os.makedirs('data/progress', exist_ok=True)
        df.to_csv(progress_file, index=False, encoding='utf-8-sig')
        print(f"\n중간 결과 저장 완료 ({current_count}/{total_rows}): {progress_file}")

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
        """레스토랑 데이터 처리 메인 함수"""
        try:
            # 데이터 로드
            df = pd.read_csv(input_file)
            total_rows = len(df)
            
            print("\n" + "="*50)
            print(f"분석 시작")
            print(f"총 처리할 레스토랑: {total_rows}개")
            print("="*50 + "\n")

            # 상태 추적 초기화
            processed = 0
            errors = 0
            start_time = time.time()
            
            # 결과 저장용 컬럼 추가
            df['review_summary'] = ''
            df['desc_summary'] = ''
            df['extracted_keywords'] = ''
            
            # 청크 단위 처리
            for i in range(0, total_rows, self.batch_size):
                chunk = df.iloc[i:min(i + self.batch_size, total_rows)]
                
                print(f"\n현재 처리중: {chunk['MCT_NM'].tolist()}")
                
                results = await self._process_batch(chunk)
                
                # 결과 저장
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        errors += 1
                        continue
                        
                    if result.get('error'):
                        errors += 1
                        continue
                        
                    row_idx = chunk.index[idx]
                    df.at[row_idx, 'review_summary'] = result.get('review_summary', '')
                    df.at[row_idx, 'desc_summary'] = result.get('desc_summary', '')
                    df.at[row_idx, 'extracted_keywords'] = result.get('keywords', '')
                    processed += 1
                
                # 진행상황 출력
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / processed if processed > 0 else 0
                estimated_remaining = avg_time_per_item * (total_rows - processed)
                
                print(f"\n진행상황:")
                print(f"처리완료: {processed}/{total_rows} ({processed/total_rows*100:.1f}%)")
                print(f"에러: {errors}개")
                print(f"평균 처리시간: {avg_time_per_item:.1f}초/건")
                print(f"예상 남은시간: {estimated_remaining/60:.1f}분")
                
                # 샘플 결과 출력
                self.print_sample_results(results, chunk)
                
                # 중간 결과 저장
                if processed % self.progress_save_interval == 0:
                    self.save_progress(df, processed, total_rows)
                
                await asyncio.sleep(1)
            
            # 최종 결과 저장
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
                            model="gpt-3.5-turbo-0125",
                            messages=[
                                {"role": "system", "content": "You are a restaurant analysis expert. Respond in Korean."},
                                {"role": "user", "content": content}
                            ],
                            temperature=0.3,
                            max_tokens=200
                        )
                        
                        # 응답 파싱
                        text = response.choices[0].message.content.strip()
                        parts = {}
                        
                        current_key = None
                        current_value = []
                        
                        # 각 줄별로 파싱
                        for line in text.split('\n'):
                            line = line.strip()
                            if line:
                                if ': ' in line:
                                    if current_key and current_value:
                                        parts[current_key] = ' '.join(current_value)
                                        current_value = []
                                    current_key, value = line.split(': ', 1)
                                    current_value.append(value)
                                else:
                                    if current_key:
                                        current_value.append(line)
                        
                        if current_key and current_value:
                            parts[current_key] = ' '.join(current_value)
                        
                        # 키워드 정리
                        keywords = []
                        for key in ['VIEW', 'INTERIOR', 'TASTE', 'SPECIALTY']:
                            if key in parts:
                                keywords.extend([k.strip() for k in parts[key].split(',')])
                        
                        return {
                            'review_summary': parts.get('REVIEW_SUMMARY', ''),
                            'desc_summary': parts.get('DESC_SUMMARY', ''),
                            'keywords': '|'.join(filter(None, keywords)),
                            'error': None
                        }
                        
                    except Exception as e:
                        if "429" in str(e):  # Rate limit
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
                    'keywords': '',
                    'error': error_msg
                }
        
        return await asyncio.gather(
            *[process_row(row) for _, row in batch.iterrows()]
        )

async def main():
    try:
        # API 키 로드
        with open('secrets.json') as f:
            config = json.load(f)
        
        # 분석기 초기화 및 실행
        analyzer = RestaurantAnalyzer(api_key=config['OPENAI_API_KEY'])
        
        # 데이터 처리
        await analyzer.process_restaurants()
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())