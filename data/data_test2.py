import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data/naver_final.csv', encoding='cp949')

# 리뷰와 설명 데이터 현황 파악
total_rows = len(df)

# 빈 리뷰/설명 개수 계산
empty_review = df['REVIEW'].apply(lambda x: x == '[]').sum()
empty_desc = df['DESC'].isna().sum()

# 리뷰/설명이 모두 없는 케이스
both_empty = df[
   (df['REVIEW'].apply(lambda x: x == '[]')) & 
   (df['DESC'].isna())
].shape[0]

print(f"전체 가게 수: {total_rows}")
print(f"리뷰 없는 가게 수: {empty_review}")
print(f"설명(DESC) 없는 가게 수: {empty_desc}")
print(f"리뷰와 설명 모두 없는 가게 수: {both_empty}")
print(f"리뷰 있는 가게 수: {total_rows - empty_review}")
print(f"설명(DESC) 있는 가게 수: {total_rows - empty_desc}")

# 데이터 정리
def clean_list_string(text):
   if pd.isna(text) or text == '[]':
       return []
   # 문자열로 된 리스트를 실제 리스트로 변환
   try:
       # 작은따옴표를 큰따옴표로 변환하여 JSON 형식으로 만듦
       text = text.replace("'", '"')
       return eval(text)
   except:
       return []

# REVIEW, KEYWORD 컬럼 정리
df_cleaned = df.copy()
df_cleaned['REVIEW'] = df_cleaned['REVIEW'].apply(clean_list_string)
df_cleaned['KEYWORD'] = df_cleaned['KEYWORD'].apply(clean_list_string)

# MENU 컬럼 정리 (가격 형식 통일)
def clean_menu(menu_text):
   if pd.isna(menu_text):
       return {}
   menu_dict = {}
   items = menu_text.split('|')
   for item in items:
       item = item.strip()
       if ' ' not in item:
           continue
       name, price = item.rsplit(' ', 1)
       price = price.replace(',', '').replace('원', '')
       try:
           price = int(price)
           menu_dict[name] = price
       except:
           continue
   return menu_dict

df_cleaned['MENU'] = df_cleaned['MENU'].apply(clean_menu)

# 정리된 데이터 저장
df_cleaned.to_csv('data/review_data_cleaned.csv', index=False, encoding='cp949')

print("\n처리된 데이터 샘플:")
print(df_cleaned.head())