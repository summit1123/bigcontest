import pandas as pd
import time
import random
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    ElementClickInterceptedException,
    WebDriverException
)
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import logging
import sys

# **로깅 설정**
logging.basicConfig(
    filename='crawler_debug.log',
    level=logging.DEBUG,  # DEBUG 레벨로 상세 로그 기록
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# **사용자 에이전트 및 브라우저 옵션 설정**
ua = UserAgent()
options = webdriver.ChromeOptions()
options.add_argument(f'user-agent={ua.random}')
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
# 헤드리스 모드 비활성화 (디버깅을 위해)
# options.add_argument('--headless')

# **프록시 설정 (필요 시 사용)**
# proxy = "http://your_proxy_address:port"
# options.add_argument(f'--proxy-server={proxy}')

# **웹드라이버 초기화 함수**
def initialize_webdriver():
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        logging.info("웹드라이버 초기화 성공")
        return driver
    except WebDriverException as e:
        logging.error(f"웹드라이버 초기화 실패: {e}")
        sys.exit(1)  # 드라이버 초기화 실패 시 스크립트 종료

driver = initialize_webdriver()

# **데이터 로드**
file_path = 'data/naver_final.csv'

try:
    df = pd.read_csv(file_path, encoding='cp949')
    logging.info("CSV 파일 로드 성공")
except FileNotFoundError:
    logging.error("CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    driver.quit()
    sys.exit(1)
except UnicodeDecodeError as e:
    logging.error(f"CSV 파일 인코딩 오류: {e}")
    driver.quit()
    sys.exit(1)
except Exception as e:
    logging.error(f"CSV 파일 로드 중 오류 발생: {e}")
    driver.quit()
    sys.exit(1)

# **REVIEW 필드가 '[]'인 행만 필터링**
filtered_df = df[df['REVIEW'].astype(str) == '[]'].reset_index(drop=True)
total_restaurants = len(filtered_df)
logging.info(f"총 {total_restaurants}개의 식당에서 리뷰를 수집합니다.")
print(f"총 {total_restaurants}개의 식당에서 리뷰를 수집합니다.")

# **리뷰 수집 함수 with 리트라이 로직**
def collect_reviews(place_url, max_reviews=5, retries=3):
    reviews = []
    attempt = 0
    while attempt < retries:
        try:
            driver.get(place_url)
            # 페이지 로드가 완료될 때까지 최대 15초 대기
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.place_section'))
            )
            logging.debug(f"페이지 로드 성공: {place_url}")
            time.sleep(random.uniform(2, 4))

            # 리뷰 더보기 버튼 클릭하여 추가 리뷰 로드
            while len(reviews) < max_reviews:
                try:
                    more_button = driver.find_element(By.CSS_SELECTOR, 'a._3Yw7k')
                    more_button.click()
                    logging.debug("더보기 버튼 클릭")
                    time.sleep(random.uniform(1, 2))
                except NoSuchElementException:
                    logging.debug("더보기 버튼 없음")
                    break
                except ElementClickInterceptedException:
                    logging.debug("더보기 버튼 클릭 방해")
                    break

                # 현재까지 로드된 리뷰 수 확인
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                reviews_elements = soup.select('li.GDXUq')
                if len(reviews_elements) >= max_reviews:
                    break

            # 리뷰 수집
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            reviews_elements = soup.select('li.GDXUq')[:max_reviews]
            for review in reviews_elements:
                try:
                    review_text = review.select_one('span.place_bluelink').get_text(strip=True)
                    rating = review.select_one('em.YEtwt').get_text(strip=True)
                    reviews.append({'rating': rating, 'review': review_text})
                    logging.debug(f"리뷰 수집: {rating} - {review_text}")
                except AttributeError:
                    logging.debug("리뷰 텍스트 또는 평점 추출 실패")
                    continue
            # 성공적으로 리뷰를 수집했으므로 루프 종료
            break
        except TimeoutException:
            attempt += 1
            logging.warning(f"페이지 로딩 시간 초과: {place_url} (리트라이 {attempt}/{retries})")
            if attempt < retries:
                time.sleep(random.uniform(5, 10))  # 리트라이 전 대기
            else:
                logging.error(f"리뷰 수집 실패: {place_url}")
        except Exception as e:
            logging.error(f"리뷰 수집 중 오류 발생: {e} - URL: {place_url}")
            break
    return reviews

# **식당 URL 수집 및 리뷰 업데이트**
failed_restaurants = []

for i, row in filtered_df.iterrows():
    keyword = row['KEYWORD']
    if not keyword or keyword.strip() == '':
        logging.warning(f"식당 '{row['NAME']}'의 키워드가 비어있습니다. 스킵합니다.")
        print(f"식당 '{row['NAME']}'의 키워드가 비어있습니다. 스킵합니다.")
        continue  # 빈 키워드인 경우 스킵

    try:
        search_url = f'https://map.naver.com/v5/search/{keyword}/place'
        driver.get(search_url)
        # 검색 결과가 로드될 때까지 최대 15초 대기
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ul._1IfMK'))
        )
        logging.debug(f"검색 페이지로 이동: {search_url}")

        # 대기 후 첫 번째 검색 결과 클릭
        time.sleep(random.uniform(2, 4))
        first_place = driver.find_element(By.CSS_SELECTOR, 'ul._1IfMK li:first-child a')
        first_place.click()
        logging.debug("첫 번째 식당 클릭")
        time.sleep(random.uniform(2, 4))

        # 현재 URL에서 restaurant ID 추출
        current_url = driver.current_url
        res_code_search = re.search(r'/restaurant/(\d+)/review/', current_url)
        if res_code_search:
            res_code = res_code_search.group(1)
            final_url = f'https://pcmap.place.naver.com/restaurant/{res_code}/review/visitor#/'
            reviews = collect_reviews(final_url, max_reviews=5, retries=3)
            if reviews:
                df.at[i, 'REVIEW'] = str(reviews)  # 리뷰 리스트를 문자열로 저장
                logging.info(f"{row['NAME']} 리뷰 수집 완료: {len(reviews)}개")
                print(f"{row['NAME']} 리뷰 수집 완료: {len(reviews)}개")
            else:
                failed_restaurants.append(row['NAME'])
                df.at[i, 'REVIEW'] = '[]'  # 실패 시 REVIEW 필드 유지
                logging.warning(f"{row['NAME']} 리뷰 수집 실패: 시간이 너무 오래 걸림")
                print(f"{row['NAME']} 리뷰 수집 실패: 시간이 너무 오래 걸림")
        else:
            # URL에서 restaurant ID 추출 실패 시, 페이지 소스에서 메타 태그로 시도
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            res_code_tags = soup.select('meta[itemprop="identifier"]')
            if res_code_tags:
                res_code = res_code_tags[0].get('content', '')
                final_url = f'https://pcmap.place.naver.com/restaurant/{res_code}/review/visitor#/'
                reviews = collect_reviews(final_url, max_reviews=5, retries=3)
                if reviews:
                    df.at[i, 'REVIEW'] = str(reviews)
                    logging.info(f"{row['NAME']} 리뷰 수집 완료: {len(reviews)}개")
                    print(f"{row['NAME']} 리뷰 수집 완료: {len(reviews)}개")
                else:
                    failed_restaurants.append(row['NAME'])
                    df.at[i, 'REVIEW'] = '[]'
                    logging.warning(f"{row['NAME']} 리뷰 수집 실패: 시간이 너무 오래 걸림")
                    print(f"{row['NAME']} 리뷰 수집 실패: 시간이 너무 오래 걸림")
            else:
                failed_restaurants.append(row['NAME'])
                logging.warning(f"{row['NAME']}의 리뷰 URL을 찾을 수 없습니다.")
                print(f"{row['NAME']}의 리뷰 URL을 찾을 수 없습니다.")
    except TimeoutException:
        failed_restaurants.append(row['NAME'])
        logging.warning(f"검색 페이지 로딩 시간 초과: {search_url}")
        print(f"{row['NAME']} 리뷰 수집 실패: 페이지 로딩 시간 초과")
    except NoSuchElementException:
        failed_restaurants.append(row['NAME'])
        logging.warning(f"{row['NAME']}의 요소를 찾을 수 없습니다.")
        print(f"{row['NAME']} 리뷰 수집 실패: 요소를 찾을 수 없습니다.")
    except Exception as e:
        failed_restaurants.append(row['NAME'])
        logging.error(f"{row['NAME']} 리뷰 수집 실패: {e}")
        print(f"{row['NAME']} 리뷰 수집 실패: {e}")
    finally:
        driver.switch_to.default_content()
        time.sleep(random.uniform(1, 3))

    # **진행 상황 저장 (중간에 크롤러가 중단될 경우 대비)**
    if (i + 1) % 100 == 0:
        try:
            df.to_csv('data/naver_final_updated.csv', index=False, encoding='cp949')
            logging.info(f"{i + 1}개의 식당 리뷰가 저장되었습니다.")
            print(f"{i + 1}개의 식당 리뷰가 저장되었습니다.")
        except Exception as e:
            logging.error(f"중간 저장 실패: {e}")
            print(f"중간 저장 실패: {e}")

# **최종 데이터 저장**
try:
    df.to_csv('data/naver_final_updated.csv', index=False, encoding='cp949')
    logging.info("리뷰 크롤링 완료 및 데이터 저장 완료.")
    print("리뷰 크롤링 완료 및 데이터 저장 완료.")
except Exception as e:
    logging.error(f"최종 데이터 저장 실패: {e}")
    print(f"최종 데이터 저장 실패: {e}")

# **실패한 식당 기록**
if failed_restaurants:
    with open('failed_restaurants.txt', 'w', encoding='cp949') as f:
        for name in failed_restaurants:
            f.write(f"{name}\n")
    logging.info(f"총 {len(failed_restaurants)}개의 식당에서 리뷰 수집 실패. 'failed_restaurants.txt'에 기록되었습니다.")
    print(f"총 {len(failed_restaurants)}개의 식당에서 리뷰 수집 실패. 'failed_restaurants.txt'에 기록되었습니다.")

driver.quit()