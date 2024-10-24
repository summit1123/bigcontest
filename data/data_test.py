import pandas as pd
import ast

def analyze_review_status(file_path: str):
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 전체 데이터 수
        total_restaurants = len(df)
        
        # REVIEW 컬럼의 데이터 형식 변환 함수
        def parse_review(x):
            if pd.isna(x):
                return []
            try:
                if isinstance(x, str):
                    parsed = ast.literal_eval(x)
                    return parsed if isinstance(parsed, list) else []
                return x if isinstance(x, list) else []
            except:
                return []
        
        # REVIEW 컬럼 파싱
        df['REVIEW_parsed'] = df['REVIEW'].apply(parse_review)
        
        # 리뷰 개수 계산
        df['review_count'] = df['REVIEW_parsed'].apply(len)
        
        # 통계 계산
        review_stats = {
            '전체 식당 수': total_restaurants,
            '리뷰 있는 식당 수': len(df[df['review_count'] > 0]),
            '리뷰 없는 식당 수': len(df[df['review_count'] == 0]),
            '전체 리뷰 수': df['review_count'].sum(),
            '식당당 평균 리뷰 수': df['review_count'].mean(),
            '최대 리뷰 수': df['review_count'].max()
        }
        
        # 리뷰 개수 분포
        review_distribution = df['review_count'].value_counts().sort_index()
        
        print("\n=== 리뷰 데이터 현황 ===")
        for key, value in review_stats.items():
            print(f"{key}: {value:,.2f}")
            
        print("\n=== 리뷰 개수 분포 ===")
        print(review_distribution)
        
        # 리뷰가 가장 많은 식당 확인
        max_reviews = df.nlargest(5, 'review_count')[['NAME', 'ADDR', 'review_count']]
        print("\n=== 리뷰가 가장 많은 식당 Top 5 ===")
        print(max_reviews)
        
        return df
        
    except Exception as e:
        print(f"Error analyzing reviews: {str(e)}")

# 실행
df = analyze_review_status('data/review_data_updated.csv')