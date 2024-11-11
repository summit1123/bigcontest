import pandas as pd

def process_restaurant_data():
    """
    원본 데이터와 업데이트 데이터를 병합하고 새로운 파일로 저장하는 함수
    """
    try:
        # 파일 경로 설정
        original_file = 'data/final_restaurant_data.csv'
        updated_file = 'data/review_data_updated_analyzed.csv'
        output_file = 'data/final_restaurant_data_v2.csv'
        
        print("데이터 파일 읽는 중...")
        
        # 원본 데이터 읽기
        try:
            print("utf-8-sig로 시도...")
            df_original = pd.read_csv(original_file, encoding='utf-8-sig')
        except:
            try:
                print("cp949로 시도...")
                df_original = pd.read_csv(original_file, encoding='cp949')
            except:
                print("utf-8로 시도...")
                df_original = pd.read_csv(original_file, encoding='utf-8')
        
        print(f"원본 데이터 레코드 수: {len(df_original)}")
        
        # 업데이트할 데이터 읽기
        df_updated = pd.read_csv(updated_file, encoding='utf-8-sig')
        print(f"업데이트 데이터 레코드 수: {len(df_updated)}")
        
        # 필요한 컬럼만 선택
        update_columns = ['MCT_NM', 'review_summary', 'desc_summary', 'keywords']
        df_updated = df_updated[update_columns]
        
        # 데이터 병합
        df_result = pd.merge(df_original, df_updated, on='MCT_NM', how='left')
        
        # 제거할 컬럼 리스트
        columns_to_remove = ['REVIEW_SUMMARY', 'DESC_SUMMARY', 'VIEW', 'INTERIOR', 'TASTE', 'SPECIALTY']
        
        # 원본 데이터에서 해당 컬럼들 제거
        existing_columns = [col for col in columns_to_remove if col in df_result.columns]
        df_result = df_result.drop(columns=existing_columns)
        
        # 결과 확인 및 저장
        print("\n처리 결과:")
        print(f"제거된 컬럼: {existing_columns}")
        print(f"남은 컬럼 수: {len(df_result.columns)}")
        print(f"남은 컬럼 목록: {', '.join(df_result.columns)}")
        
        # 데이터 샘플 확인
        print("\n첫 번째 레코드 샘플:")
        print(df_result.iloc[0])
        
        # 결과 파일 저장
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n처리된 데이터가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        print("스택 트레이스:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_restaurant_data()