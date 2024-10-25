import pandas as pd

# 허용된 키워드 정의
VALID_KEYWORDS = {
    'VIEW': {
        '바다 전망',
        '일몰 뷰',
        '산 전망',
        '-'
    },
    'INTERIOR': {
        '모던한',
        '빈티지',
        '전통적인',
        '고급스러운',
        '캐주얼한',
        '로맨틱한',
        '-'
    },
    'TASTE': {
        '매운맛',
        '진한맛',
        '담백한',
        '달콤한',
        '감칠맛',
        '-'
    },
    'SPECIALTY': {
        '제주 식재료',
        '계절 메뉴',
        '수제/직접만든',
        '-'
    }
}

def validate_and_clean_keywords():
    try:
        # 데이터 로드
        df = pd.read_csv('data/final_merged_data_fixed.csv', encoding='cp949')
        
        print(f"전체 데이터 수: {len(df)}")
        
        # 각 카테고리별 키워드 검증 및 정리
        stats = {}
        for category in VALID_KEYWORDS.keys():
            invalid_count = 0
            valid_count = 0
            english_count = 0
            
            def clean_keyword(keyword):
                nonlocal invalid_count, valid_count, english_count
                
                if pd.isna(keyword) or keyword == '':
                    invalid_count += 1
                    return '-'
                    
                # 영어 키워드 체크 (영어 알파벳 포함 여부)
                if any(c.isascii() and c.isalpha() for c in keyword):
                    english_count += 1
                    return '-'
                
                # 유효한 키워드인지 체크
                if keyword in VALID_KEYWORDS[category]:
                    valid_count += 1
                    return keyword
                else:
                    invalid_count += 1
                    return '-'
            
            # 키워드 정리
            df[category] = df[category].apply(clean_keyword)
            
            # 통계 저장
            stats[category] = {
                'valid': valid_count,
                'invalid': invalid_count,
                'english': english_count,
                'total': len(df)
            }
        
        # 결과 저장
        df.to_csv('data/final_merged_data_cleaned_final.csv', index=False, encoding='cp949')
        
        # 결과 출력
        print("\n=== 카테고리별 키워드 검증 결과 ===")
        for category, stat in stats.items():
            print(f"\n{category}:")
            print(f"- 전체 데이터: {stat['total']:,}개")
            print(f"- 유효한 키워드: {stat['valid']:,}개 ({stat['valid']/stat['total']*100:.1f}%)")
            print(f"- 유효하지 않은 키워드: {stat['invalid']:,}개 ({stat['invalid']/stat['total']*100:.1f}%)")
            print(f"- 영어 키워드: {stat['english']:,}개 ({stat['english']/stat['total']*100:.1f}%)")
            
            print("\n현재 키워드 분포:")
            value_counts = df[category].value_counts()
            print(value_counts)
        
        print("\n=== 샘플 데이터 ===")
        print(df[['MCT_NM'] + list(VALID_KEYWORDS.keys())].head())
        
        return df, stats
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        df, stats = validate_and_clean_keywords()
        print("\n데이터 정리가 완료되었습니다.")
    except Exception as e:
        print(f"최종 에러: {str(e)}")