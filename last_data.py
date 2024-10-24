# import pandas as pd

# def convert_to_korean(value):
#     """영어 키워드를 한글로 변환"""
#     # 영어-한글 매핑 딕셔너리
#     translation = {
#         # VIEW
#         'ocean_view': '바다 전망',
#         'sunset_view': '일몰 뷰',
#         'mountain_view': '산 전망',
        
#         # INTERIOR
#         'modern': '모던한',
#         'vintage': '빈티지',
#         'traditional': '전통적인',
#         'luxury': '고급스러운',
#         'casual': '캐주얼한',
#         'romantic': '로맨틱한',
        
#         # TASTE
#         'spicy': '매운맛',
#         'rich': '진한맛',
#         'light': '담백한',
#         'sweet': '달콤한',
#         'savory': '감칠맛',
        
#         # SPECIALTY
#         'jeju_ingredients': '제주 식재료',
#         'seasonal': '계절 메뉴',
#         'handmade': '수제/직접만든'
#     }
    
#     if pd.isna(value) or value == '' or value == '-':
#         return '-'
    
#     # 쉼표로 구분된 키워드들 처리
#     keywords = [k.strip() for k in value.split('|')]
#     translated = []
    
#     for keyword in keywords:
#         if keyword in translation:
#             translated.append(translation[keyword])
#         else:
#             translated.append(keyword)
    
#     return '|'.join(translated)

# # 메인 실행 코드
# if __name__ == "__main__":
#     # 데이터 로드
#     df = pd.read_csv('data/review_data_updated_analyzed.csv')
    
#     # extracted_keywords 컬럼에서 영어 키워드를 한글로 변환
#     if 'extracted_keywords' in df.columns:
#         keywords = df['extracted_keywords'].apply(lambda x: str(x).split('|'))
#         for idx, kw_list in keywords.items():
#             translated = [convert_to_korean(k.strip()) if k.strip() != '' else '-' for k in kw_list]
#             df.at[idx, 'extracted_keywords'] = '|'.join(translated)
    
#     # 변환된 데이터 저장
#     df.to_csv('data/review_data_korean.csv', index=False)
    
#     # 결과 확인
#     print("\n변환 결과 샘플:")
#     for idx, row in df.head().iterrows():
#         print(f"\n가게명: {row['MCT_NM']}")
#         print(f"변환된 키워드: {row['extracted_keywords']}")

# import pandas as pd

# def merge_final_data():
#     """
#     final_coordinates.csv와 review_data_korean.csv를 병합하고 정리
#     """
#     try:
#         # encoding='cp949'로 데이터 로드 시도
#         try:
#             coords_df = pd.read_csv('data/final_coordinates.csv', encoding='cp949')
#         except:
#             coords_df = pd.read_csv('data/final_coordinates.csv', encoding='utf-8')
            
#         try:
#             review_df = pd.read_csv('data/review_data_korean.csv', encoding='cp949')
#         except:
#             review_df = pd.read_csv('data/review_data_korean.csv', encoding='utf-8')
        
#         print("데이터 로드 완료")
        
#         # 제외할 컬럼 리스트
#         columns_to_drop = ['REVIEW', 'DESC', 'CATEGORY', 'ADDR', 'KEYWORD']
        
#         # review_df에서 불필요한 컬럼 제거
#         review_df = review_df.drop(columns=[col for col in columns_to_drop if col in review_df.columns])
        
#         print("컬럼 제거 완료")
#         print("review_df 컬럼:", review_df.columns.tolist())
#         print("coords_df 컬럼:", coords_df.columns.tolist())
        
#         # MCT_NM을 기준으로 데이터 병합
#         merged_df = pd.merge(coords_df, review_df, on='MCT_NM', how='left')
        
#         print("데이터 병합 완료")
        
#         # 중복 컬럼 처리 (suffix가 있는 컬럼 제거)
#         for col in merged_df.columns:
#             if col.endswith('_x') or col.endswith('_y'):
#                 base_col = col[:-2]  # suffix 제거
#                 # _x 컬럼 유지하고 _y 컬럼 삭제
#                 if f"{base_col}_y" in merged_df.columns:
#                     merged_df = merged_df.drop(columns=[f"{base_col}_y"])
#                 merged_df = merged_df.rename(columns={f"{base_col}_x": base_col})
        
#         print("중복 컬럼 처리 완료")
        
#         # 최종 데이터 저장 (cp949로 저장)
#         merged_df.to_csv('data/final_merged_data.csv', index=False, encoding='cp949')
        
#         print("데이터 저장 완료")
        
#         return merged_df
        
#     except Exception as e:
#         print(f"처리 중 에러 발생: {str(e)}")
#         raise e

# if __name__ == "__main__":
#     try:
#         # 데이터 병합 실행
#         merged_data = merge_final_data()
        
#         # 결과 확인
#         print("\n최종 데이터 정보:")
#         print(f"데이터 크기: {merged_data.shape}")
#         print("\n포함된 컬럼:")
#         print(merged_data.columns.tolist())
        
#         # 샘플 데이터 출력
#         print("\n첫 번째 행 샘플:")
#         sample_row = merged_data.iloc[0]
#         for col in merged_data.columns:
#             print(f"{col}: {sample_row[col]}")
        
#     except Exception as e:
#         print(f"최종 에러: {str(e)}")

# import pandas as pd

# def remove_name_column():
#     """
#     final_merged_data.csv에서 NAME 컬럼 제거
#     """
#     # 데이터 로드
#     try:
#         df = pd.read_csv('data/final_merged_data.csv', encoding='cp949')
#     except:
#         df = pd.read_csv('data/final_merged_data.csv', encoding='utf-8')
    
#     print("데이터 로드 완료")
    
#     # NAME 컬럼 제거
#     if 'NAME' in df.columns:
#         df = df.drop(columns=['NAME'])
#         print("NAME 컬럼 제거 완료")
#     else:
#         print("NAME 컬럼이 데이터에 없습니다")
    
#     # 수정된 데이터 저장
#     df.to_csv('data/final_merged_data.csv', index=False, encoding='cp949')
#     print("최종 데이터 저장 완료")
    
#     # 최종 컬럼 확인
#     print("\n최종 컬럼 목록:")
#     print(df.columns.tolist())
    
#     return df

# if __name__ == "__main__":
#     try:
#         final_df = remove_name_column()
#         print(f"\n최종 데이터 크기: {final_df.shape}")
#     except Exception as e:
#         print(f"에러 발생: {str(e)}")

import pandas as pd

def check_and_fix_keywords():
    """
    데이터의 컬럼과 키워드 형식을 확인하고 수정
    """
    try:
        # 데이터 로드
        try:
            df = pd.read_csv('data/final_merged_data.csv', encoding='cp949')
        except:
            df = pd.read_csv('data/final_merged_data.csv', encoding='utf-8')
        
        # 현재 컬럼 확인
        print("현재 데이터의 컬럼:")
        print(df.columns.tolist())
        
        # 키워드 관련 컬럼 확인
        keyword_columns = [col for col in df.columns if 'KEY' in col.upper() or 'VIEW' in col.upper() 
                         or 'INTERIOR' in col.upper() or 'TASTE' in col.upper() or 'SPECIALTY' in col.upper()]
        print("\n키워드 관련 컬럼:")
        print(keyword_columns)
        
        # 첫 몇 행의 데이터 확인
        print("\n첫 번째 행의 데이터:")
        sample_row = df.iloc[0]
        for col in df.columns:
            print(f"{col}: {sample_row[col]}")
            
        return df
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        df = check_and_fix_keywords()
        
    except Exception as e:
        print(f"최종 에러: {str(e)}")