import pandas as pd
import numpy as np

# 원본 데이터 로드 - 명시적으로 cp949 인코딩 사용
df = pd.read_csv('data/old/final_tour.csv', encoding='cp949')

# 1. 텍스트 데이터 정리
def clean_text(text):
    if pd.isna(text):
        return ''
    # 줄바꿈, HTML 태그 등 정리
    text = str(text).replace('\n', ' ').replace('<br />', ' ')
    # 연속된 공백 제거
    text = ' '.join(text.split())
    return text

# 2. 각 컬럼 정리
df_cleaned = df.copy()

# 텍스트 컬럼 정리
text_columns = ['명칭', '개요', '상세정보', '체험안내', '문의 및 안내', '이용시간', '주차시설']
for col in text_columns:
    df_cleaned[col] = df_cleaned[col].apply(clean_text)

# 필수 필드 결측치 처리
df_cleaned['명칭'] = df_cleaned['명칭'].fillna('이름 없는 장소')
df_cleaned['개요'] = df_cleaned['개요'].fillna('')
df_cleaned['상세정보'] = df_cleaned['상세정보'].fillna('')

# 3. 시설 정보 추출 (주차, 화장실 등)
def extract_facilities(text):
    facilities = []
    if '주차' in text and '있음' in text:
        facilities.append('주차가능')
    if '화장실' in text and '있음' in text:
        facilities.append('화장실있음')
    return ','.join(facilities)

df_cleaned['시설정보'] = df_cleaned['상세정보'].apply(extract_facilities)

# 4. 불필요한 컬럼 제거 또는 이름 변경
columns_to_keep = [
    '명칭', '우편번호', '주소', '위도', '경도', 
    '개요', '문의 및 안내', '개장일', '쉬는날',
    '이용시간', '주차시설', '시설정보'
]

df_final = df_cleaned[columns_to_keep]

# 5. CSV 파일로 저장 - UTF-8 with BOM 사용
df_final.to_csv('data/final_tour_v2.csv', index=False, encoding='utf-8-sig')

# 저장된 파일 다시 읽어서 확인
df_check = pd.read_csv('data/final_tour_v2.csv', encoding='utf-8-sig')
print("전처리 전 shape:", df.shape)
print("전처리 후 shape:", df_final.shape)
print("\n처리된 데이터 샘플:")
print(df_check.head())