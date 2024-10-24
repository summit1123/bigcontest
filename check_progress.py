import pandas as pd
import glob
import os
from datetime import datetime

def check_progress():
    # 1. 원본 파일 확인
    try:
        original_df = pd.read_csv('data/review_data_updated.csv')
        total_rows = len(original_df)
        print(f"전체 데이터 수: {total_rows}")
    except:
        print("원본 파일을 찾을 수 없습니다.")
        total_rows = 0

    # 2. 결과 파일 찾기
    result_files = glob.glob('*with_new_categories*.csv')
    
    if not result_files:
        print("\n아직 결과 파일이 생성되지 않았습니다.")
        return
    
    # 최신 파일 확인
    latest_file = max(result_files, key=os.path.getctime)
    latest_df = pd.read_csv(latest_file)
    
    # 3. 진행 상황 분석
    processed_rows = latest_df[latest_df['new_category_view'] != ''].copy()
    processed_count = len(processed_rows)
    
    print(f"\n최신 결과 파일: {latest_file}")
    print(f"마지막 수정 시간: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")
    print(f"\n처리된 데이터 수: {processed_count}")
    
    if total_rows > 0:
        progress = (processed_count / total_rows) * 100
        print(f"전체 진행률: {progress:.2f}%")
    
    # 4. 최근 처리된 5개 항목 출력
    print("\n최근 처리된 5개 항목:")
    recent_processed = processed_rows.tail(5)
    
    for idx, row in recent_processed.iterrows():
        print(f"\n식당명: {row['MCT_NM']}")
        print("카테고리:")
        for cat in ['view', 'interior', 'taste', 'specialty', 'portion', 'occasion', 'special']:
            cat_val = row.get(f'new_category_{cat}', '')
            if cat_val and cat_val != '[]':
                print(f"- {cat}: {cat_val}")

if __name__ == "__main__":
    check_progress()
    print("\n10초 후 다시 확인합니다...")
    while True:
        import time
        time.sleep(10)  # 10초 대기
        print("\n" + "="*50)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        check_progress()