# utils/fda_realtime_crawler.py
"""
실시간 FDA 리콜 데이터 크롤링 및 업데이트 모듈 - Selenium 기반으로 수정
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_recall_chunks(text, chunk_size=800, overlap_size=120):
    """리콜 Company Announcement 텍스트를 청크로 분할 (기존 코드와 동일)"""
    
    if not text or len(text.strip()) < 100:
        return []
    
    def protect_important_info(text):
        """중요한 정보 보호 (회사명, 제품명, 날짜 등)"""
        patterns = [
            r'[A-Z][a-zA-Z\s&.,]+ LLC',  # 회사명
            r'[A-Z][a-zA-Z\s&.,]+ Inc\.',  # 회사명
            r'[A-Z][a-zA-Z\s&.,]+ Company',  # 회사명
            r'\d+\s*boxes?\s*of\s*[^,]+',  # 제품 수량
            r'Lot\s*#?\s*\d+',  # 로트 번호
            r'1-\d{3}-\d{3}-\d{4}',  # 전화번호
        ]
        
        protected_refs = {}
        protected_text = text
        
        for i, pattern in enumerate(patterns):
            matches = list(re.finditer(pattern, protected_text, re.IGNORECASE))
            for j, match in enumerate(matches):
                ref_id = f"__PROTECT_{i}_{j}__"
                protected_refs[ref_id] = match.group()
                protected_text = protected_text.replace(match.group(), ref_id, 1)
        
        return protected_text, protected_refs
    
    def restore_protected_info(text, protected_refs):
        """보호된 정보 복원"""
        for ref_id, original_text in protected_refs.items():
            text = text.replace(ref_id, original_text)
        return text
    
    # 정보 보호
    protected_text, protected_refs = protect_important_info(text)
    
    # 청크 생성
    chunks = []
    start = 0
    min_chunk_size = 150
    max_chunk_size = 1200
    
    while start < len(protected_text):
        end = start + chunk_size
        
        if end >= len(protected_text):
            chunk = protected_text[start:]
            if chunk.strip() and len(chunk.strip()) >= min_chunk_size:
                restored_chunk = restore_protected_info(chunk.strip(), protected_refs)
                chunks.append(restored_chunk)
            break
        
        # 적절한 분할점 찾기
        chunk_text = protected_text[start:end]
        split_candidates = []
        
        separators = ['. ', '.\n', ';\n', '; ', ',\n', ', ', ')\n', ') ', '\n\n', '\n', ' ']
        
        for sep in separators:
            last_sep_pos = chunk_text.rfind(sep)
            if last_sep_pos > chunk_size * 0.7:
                split_candidates.append(last_sep_pos + len(sep))
        
        if split_candidates:
            actual_end = start + max(split_candidates)
        else:
            actual_end = min(end, start + max_chunk_size)
        
        chunk = protected_text[start:actual_end].strip()
        if chunk and len(chunk) >= min_chunk_size:
            restored_chunk = restore_protected_info(chunk, protected_refs)
            chunks.append(restored_chunk)
        
        # 다음 청크 시작점 (오버랩 적용)
        start = max(actual_end - overlap_size, start + min_chunk_size)
    
    return chunks

# def get_latest_date_from_vectorstore(vectorstore):
#     """벡터스토어에서 가장 최근 날짜 조회"""
#     try:
#         all_data = vectorstore.get()
#         metadatas = all_data.get('metadatas', [])
        
#         latest_date = None
#         for metadata in metadatas:
#             if metadata and metadata.get('effective_date'):
#                 date_str = metadata['effective_date']
#                 try:
#                     date_obj = datetime.strptime(date_str, '%Y-%m-%d')
#                     if latest_date is None or date_obj > latest_date:
#                         latest_date = date_obj
#                 except:
#                     continue
        
#         if latest_date:
#             print(f"📅 벡터DB 최신 날짜: {latest_date.strftime('%Y-%m-%d')}")
#             return latest_date
#         else:
#             # 벡터DB에 날짜가 없으면 30일 전부터
#             fallback_date = datetime.now() - timedelta(days=30)
#             print(f"📅 벡터DB에 날짜 없음, 기본값 사용: {fallback_date.strftime('%Y-%m-%d')}")
#             return fallback_date
            
#     except Exception as e:
#         print(f"최신 날짜 조회 오류: {e}")
#         return datetime.now() - timedelta(days=30)

# utils/fda_realtime_crawler.py 의 FDARealtimeCrawler 클래스 수정

class FDARealtimeCrawler:
    def __init__(self):
        self.base_url = "https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
        self.driver = None
        
    def _init_driver(self):
        """이 코드는 최적화된 FDA 크롤링 드라이버를 초기화합니다"""
        if self.driver is None:
            service = Service(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-logging')
            options.add_argument('--log-level=3')
            options.add_argument('--page-load-strategy=normal')  # FDA는 JS 필요
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(15)  # 타임아웃 단축

    def _close_driver(self):
        """드라이버 종료 - 누락된 메서드 추가"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"드라이버 종료 중 오류: {e}")
            finally:
                self.driver = None

    def check_food_beverages_in_summary(self, url):
        try:
            self.driver.get(url)
            time.sleep(1.0)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Summary 섹션에서 Product Type만 확인
            summary_section = soup.find('h2', string='Summary')
            if not summary_section:
                print(f"      ❌ Summary 섹션을 찾을 수 없음")
                return False
            
            summary_content = summary_section.find_next('div', class_='inset-column')
            if not summary_content:
                print(f"      ❌ Summary 내용을 찾을 수 없음")
                return False
            
            # Product Type 찾기
            for dt in summary_content.find_all('dt'):
                if 'Product Type' in dt.get_text():
                    dd = dt.find_next('dd')
                    if dd:
                        product_type = dd.get_text().strip()
                        print(f"      🔍 Product Type: '{product_type}'")
                        
                        # 정확히 "Food & Beverages" 포함 여부만 확인
                        if 'Food & Beverages' in product_type:
                            print(f"      ✅ Food & Beverages 확인됨")
                            return True
                        else:
                            print(f"      ❌ Food & Beverages 아님")
                            return False
            
            print(f"      ❌ Product Type 필드를 찾을 수 없음")
            return False
            
        except Exception as e:
            print(f"      ❌ Product Type 확인 중 오류: {e}")
            return False

    def get_existing_urls_from_vectorstore(self, vectorstore):
        try:
            existing_data = vectorstore.get()
            existing_urls = set()
            for metadata in existing_data.get('metadatas', []):
                if metadata and 'url' in metadata:
                    existing_urls.add(metadata['url'])
            print(f"📋 기존 벡터DB URL: {len(existing_urls)}개")
            return existing_urls
        except Exception as e:
            print(f"기존 URL 확인 오류: {e}")
            return set()
    
    def crawl_latest_recalls(self, days_back: int = 7) -> List[Dict]:
        """최근 N일간의 리콜 데이터 크롤링"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recalls = []
        
        try:
            self._init_driver()
            
            self.driver.get(self.base_url)
            time.sleep(1)
            
            print("🎯 Food & Beverages 필터 적용 중...")
            
            # Product Type 드롭다운 클릭
            try:
                product_type_dropdown = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "select[name='field_regulated_product_field']"))
                )
                product_type_dropdown.click()
                time.sleep(1)
                
                # Food & Beverages 옵션 선택
                food_beverages_option = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "option[value='2323']"))
                )
                food_beverages_option.click()
                time.sleep(1)
                
                print("✅ Food & Beverages 필터 적용 완료")
                
            except Exception as e:
                print(f"❌ 필터 적용 실패: {e}")
                return []
            
            # 필터 적용 후 테이블 로딩 대기
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "datatable"))
                )
                time.sleep(2)
                print("📊 필터링된 테이블 로딩 완료")
            except TimeoutException:
                print("❌ 필터링된 테이블 로딩 실패")
                return []
            
            print(f"📅 수집 기준: {cutoff_date.strftime('%Y-%m-%d')} 이후 데이터만 수집")
            
            processed_urls = set()  # 현재 세션 중복 방지용
            max_pages = 5  # 페이지 범위 확장
            found_old_data = False  # 오래된 데이터 발견 시 중단용
            
            for page in range(1, max_pages + 1):
                print(f"페이지 {page} 처리 중...")
                
                # 현재 페이지의 리콜 링크 및 날짜 수집
                page_recall_data = []
                try:
                    table = self.driver.find_element(By.ID, "datatable")
                    rows = table.find_elements(By.XPATH, ".//tbody/tr")
                    
                    for row in rows:
                        try:
                            # 날짜 정보 추출
                            date_cell = row.find_elements(By.TAG_NAME, "td")[0]
                            date_text = date_cell.text.strip()
                            
                            # Brand Name 링크 추출
                            brand_cell = row.find_elements(By.TAG_NAME, "td")[1]
                            link_element = brand_cell.find_element(By.TAG_NAME, "a")
                            recall_url = link_element.get_attribute('href')
                            
                            # 🆕 테이블에서 날짜 파싱
                            try:
                                # "06/30/2025" 형태를 datetime으로 변환
                                table_date = datetime.strptime(date_text, '%m/%d/%Y')
                                
                                print(f"      발견: {date_text} ({table_date.strftime('%Y-%m-%d')}) - {recall_url[-40:]}...")
                                
                                # 🆕 날짜 기반 필터링
                                if table_date >= cutoff_date:
                                    # 현재 세션 중복 체크만 수행
                                    if recall_url not in processed_urls:
                                        page_recall_data.append({
                                            'url': recall_url,
                                            'table_date': table_date,
                                            'date_text': date_text
                                        })
                                        processed_urls.add(recall_url)
                                        print(f"      ✅ 수집 대상: {date_text}")
                                    else:
                                        print(f"      ⏩ 현재 세션 중복: {date_text}")
                                else:
                                    print(f"      ❌ 날짜 필터링: {date_text} (기준: {cutoff_date.strftime('%Y-%m-%d')} 이후)")
                                    found_old_data = True
                                    
                            except ValueError as e:
                                print(f"      ⚠️ 날짜 파싱 실패: {date_text} - {e}")
                                continue
                                
                        except Exception as e:
                            print(f"      ❌ 행 처리 오류: {e}")
                            continue
                    
                    print(f"페이지 {page}: 수집 대상 {len(page_recall_data)}개 발견")
                    
                except Exception as e:
                    print(f"페이지 {page} 링크 수집 오류: {e}")
                    break
                
                # 🆕 수집 대상이 없으면서 오래된 데이터를 발견했으면 중단
                if not page_recall_data and found_old_data:
                    print("오래된 데이터 발견으로 크롤링 중단")
                    break
                
                # 개별 리콜 페이지 처리
                for i, recall_info in enumerate(page_recall_data[:10]):
                    try:
                        recall_url = recall_info['url']
                        print(f"  데이터 추출 중 ({i+1}/{min(10, len(page_recall_data))}): {recall_info['date_text']}...")
                        
                        # 메타데이터 추출
                        recall_data = self.extract_recall_metadata_direct(recall_url)
                        
                        if recall_data:
                            print(f"    ✅ 수집 완료: {recall_data['title'][:40]}... (날짜: {recall_data.get('effective_date', 'N/A')})")
                            recalls.append(recall_data)
                            
                            # 충분히 수집했으면 중단
                            if len(recalls) >= 10:
                                print(f"목표 달성: {len(recalls)}건 수집 완료")
                                return recalls
                        
                    except Exception as e:
                        print(f"    ❌ 데이터 추출 오류: {e}")
                        continue
                
                # 다음 페이지로 이동
                if page < max_pages and page_recall_data:  # 수집 데이터가 있을 때만 다음 페이지
                    try:
                        next_button = WebDriverWait(self.driver, 8).until(
                            EC.element_to_be_clickable((By.ID, "datatable_next"))
                        )
                        
                        if "disabled" in (next_button.get_attribute("class") or ""):
                            print("마지막 페이지 도달")
                            break
                        
                        next_link = next_button.find_element(By.TAG_NAME, "a")
                        self.driver.execute_script("arguments[0].click();", next_link)
                        time.sleep(1.5)
                        
                    except Exception as e:
                        print(f"페이지 이동 오류: {e}")
                        break
                else:
                    print("수집할 데이터가 없어 크롤링 종료")
                    break
            
            print(f"크롤링 완료: {cutoff_date.strftime('%Y-%m-%d')} 이후 Food & Beverages 리콜 {len(recalls)}건 수집")
            return recalls
            
        except Exception as e:
            print(f"크롤링 전체 오류: {e}")
            return []
            
        finally:
            self._close_driver()

    

    def extract_recall_metadata(self, url):
        """메타데이터 추출 - 청크 없이 전체 내용 저장"""
        try:
            self.driver.get(url)
            time.sleep(1)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
            # 제목, 날짜 추출 (기존과 동일)
            title = ""
            title_selectors = ['h1.content-title', 'h1[class*="content-title"]', 'h1']
            
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text().strip()
                    break
            
            # 날짜 정보 추출 (기존과 동일)
            effective_date = ""
            last_updated = ""
            
            def parse_date_text(date_text):
                if not date_text:
                    return ""
                try:
                    date_obj = datetime.strptime(date_text.strip(), '%B %d, %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    return ""
            
            # Summary 섹션에서 날짜 추출 (기존과 동일)
            summary_section = soup.find('h2', string='Summary')
            if summary_section:
                summary_content = summary_section.find_next('div', class_='inset-column')
                if summary_content:
                    for dt in summary_content.find_all('dt'):
                        if 'Company Announcement Date' in dt.get_text():
                            dd = dt.find_next('dd')
                            if dd:
                                time_element = dd.find('time')
                                if time_element:
                                    effective_date = parse_date_text(time_element.get_text())
                                break
                    
                    for dt in summary_content.find_all('dt'):
                        if 'FDA Publish Date' in dt.get_text():
                            dd = dt.find_next('dd')
                            if dd:
                                time_element = dd.find('time')
                                if time_element:
                                    last_updated = parse_date_text(time_element.get_text())
                                break
            
            # 🆕 Company Announcement 전체 내용 저장 (청크 없이)
            company_announcement = ""
            announcement_section = soup.find('h2', string='Company Announcement')
            if announcement_section:
                current = announcement_section.find_next_sibling()
                announcement_parts = []
                
                while current and current.name != 'hr':
                    if current.name == 'p':
                        text = current.get_text().strip()
                        if text:
                            announcement_parts.append(text)
                    current = current.find_next_sibling()
                
                company_announcement = '\n\n'.join(announcement_parts)
        
            return {
                "document_type": "recall",
                "category": "Food & Beverages", 
                "title": title,
                "url": url,
                "effective_date": effective_date,
                "last_updated": last_updated,
                "full_content": company_announcement  # 🆕 청크 대신 전체 내용
            }
        except Exception as e:
            print(f"메타데이터 추출 오류: {e}")
            return None

        
    def extract_recall_metadata_direct(self, url):
        """Food & Beverages 확인 없이 바로 메타데이터 추출"""
        return self.extract_recall_metadata(url)  # 기존 메서드 재사용

def update_vectorstore_with_new_data(new_recalls: List[Dict], vectorstore) -> int:
    """새로운 리콜 데이터를 벡터스토어에 추가 - 청크 없이 단일 문서로"""
    if not new_recalls or not vectorstore:
        print("⚠️ 추가할 데이터가 없거나 벡터스토어가 없습니다")
        return 0
    
    try:
        # 기존 중복 확인 (동일)
        existing_urls = set()
        try:
            existing_data = vectorstore.get()
            for metadata in existing_data.get('metadatas', []):
                if metadata and 'url' in metadata:
                    existing_urls.add(metadata['url'])
            print(f"📋 기존 벡터스토어: {len(existing_urls)}개 URL 확인")
        except Exception as e:
            print(f"기존 데이터 확인 중 오류: {e}")
        
        # 🆕 새 문서들 생성 (1개 URL = 1개 문서)
        new_documents = []
        processed_urls = set()
        
        for recall in new_recalls:
            recall_url = recall.get('url', '')
            
            # 중복 체크
            if recall_url in existing_urls or recall_url in processed_urls:
                print(f"⏩ 중복 건너뛰기: {recall.get('title', '')[:50]}...")
                continue
            
            processed_urls.add(recall_url)
            
            # 🆕 전체 내용 처리 (청크 없이)
            full_content = recall.get("full_content", "")
            if not full_content or len(full_content.strip()) < 100:
                print(f"⚠️ 내용 부족: {recall.get('title', '')}")
                continue
            
            # 구조화된 컨텐츠 생성
            structured_content = f"""
제목: {recall.get('title', '')}
카테고리: {recall.get('category', '')}
등급: {recall.get('class', 'Unclassified')}
발효일: {recall.get('effective_date', '')}
최종 업데이트: {recall.get('last_updated', '')}

리콜 내용:
{full_content}
            """.strip()
            
            # 🆕 메타데이터 (chunk_index 제거)
            metadata = {
                "document_type": "recall",
                "category": recall.get('category', ''),
                "class": recall.get('class', 'Unclassified'),
                "title": recall.get('title', ''),
                "url": recall_url,
                "effective_date": recall.get('effective_date', ''),
                "last_updated": recall.get('last_updated', ''),
                "source": "realtime_crawl",
                "crawl_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            doc = Document(page_content=structured_content, metadata=metadata)
            new_documents.append(doc)
        
        # 벡터스토어에 추가 (동일)
        if new_documents:
            try:
                vectorstore.add_documents(new_documents)
                print(f"✅ 벡터스토어에 {len(new_documents)}개 새 문서 추가 완료")
                
                total_count = vectorstore._collection.count()
                print(f"📊 현재 벡터스토어 총 문서 수: {total_count}개")
                
            except Exception as e:
                print(f"❌ 벡터스토어 추가 오류: {e}")
                return 0
        else:
            print("ℹ️ 추가할 새 문서가 없습니다")
            
        return len(new_documents)
        
    except Exception as e:
        print(f"❌ 벡터스토어 업데이트 전체 오류: {e}")
        return 0

# 벡터스토어 상태 확인 함수 추가
def check_vectorstore_status(vectorstore=None) -> Dict[str, Any]:
    """벡터스토어 현재 상태 확인"""
    if vectorstore is None:
        from utils.chat_recall import recall_vectorstore
        vectorstore = recall_vectorstore
    
    if vectorstore is None:
        return {
            "status": "disconnected",
            "error": "벡터스토어에 연결할 수 없습니다"
        }
    
    try:
        # 컬렉션 정보 가져오기
        collection_data = vectorstore.get()
        metadatas = collection_data.get('metadatas', [])
        
        # 실시간 데이터 카운트
        realtime_count = sum(1 for m in metadatas if m and m.get('source') == 'realtime_crawl')
        total_count = len(metadatas)
        
        # 최근 크롤링 시간
        recent_crawl = None
        for metadata in metadatas:
            if metadata and metadata.get('source') == 'realtime_crawl':
                crawl_time = metadata.get('crawl_timestamp')
                if crawl_time and (recent_crawl is None or crawl_time > recent_crawl):
                    recent_crawl = crawl_time
        
        return {
            "status": "connected",
            "total_documents": total_count,
            "realtime_documents": realtime_count,
            "database_documents": total_count - realtime_count,
            "realtime_ratio": (realtime_count / total_count * 100) if total_count > 0 else 0,
            "recent_crawl_time": recent_crawl,
            "vectorstore_path": vectorstore._persist_directory if hasattr(vectorstore, '_persist_directory') else 'Unknown'
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"상태 확인 오류: {e}"
        }
    
def get_latest_crawl_time(df):
    """최근 크롤링 시간 반환"""
    try:
        realtime_df = df[df['is_realtime'] == True]
        if realtime_df.empty:
            return "없음"
        
        crawl_times = realtime_df['crawl_timestamp'].dropna()
        if crawl_times.empty:
            return "없음"
        
        latest_time = crawl_times.max()
        return latest_time if latest_time else "없음"
        
    except Exception:
        return "없음"
    
    
# 나머지 시각화 및 유틸리티 함수들은 기존과 동일하게 유지
def create_recall_visualizations(vectorstore) -> Dict[str, Any]:
    """리콜 데이터 시각화 생성 - 실시간 데이터 구분 표시"""
    try:
        all_data = vectorstore.get()
        metadatas = all_data.get('metadatas', [])
        documents = all_data.get('documents', [])
        
        if not metadatas:
            return {}
        
        # 데이터프레임 생성 (실시간 데이터 구분)
        df_data = []
        for i, metadata in enumerate(metadatas):
            if metadata:
                doc_content = documents[i] if i < len(documents) else ""
                is_realtime = metadata.get('source') == 'realtime_crawl'
                
                df_data.append({
                    'category': metadata.get('category', 'Other') or 'Other',
                    'class': metadata.get('class', 'Unclassified') or 'Unclassified',
                    'effective_date': metadata.get('effective_date', '') or '',
                    'source': metadata.get('source', 'unknown') or 'unknown',
                    'is_realtime': is_realtime,  # 실시간 데이터 여부
                    'title': metadata.get('title', '') or '',
                    'content': doc_content[:500],
                    'url': metadata.get('url', '') or '',
                    'crawl_timestamp': metadata.get('crawl_timestamp', '') or ''
                })
        
        if not df_data:
            return {}
            
        df = pd.DataFrame(df_data)
        
        # 실시간 데이터 통계
        realtime_count = len(df[df['is_realtime'] == True])
        total_count = len(df)
        database_count = total_count - realtime_count
        
        # 통계 요약 (실시간 데이터 정보 추가)
        stats = {
            'total_recalls': total_count,
            'realtime_recalls': realtime_count,
            'database_recalls': database_count,
            'realtime_ratio': (realtime_count / total_count * 100) if total_count > 0 else 0,
            'date_range': get_date_range(df),
            'avg_monthly': calculate_monthly_average(df),
            'peak_month': get_peak_month(df),
            'latest_crawl': get_latest_crawl_time(df)
        }
        
        return {
            'stats': stats,
            'dataframe': df.drop(['content'], axis=1)
        }
        
    except Exception as e:
        print(f"시각화 생성 오류: {e}")
        return {}

def get_date_range(df):
    """날짜 범위 계산"""
    try:
        dates = pd.to_datetime(df['effective_date'], errors='coerce').dropna()
        if dates.empty:
            return "N/A"
        return f"{dates.min().strftime('%Y-%m')} ~ {dates.max().strftime('%Y-%m')}"
    except:
        return "N/A"

def calculate_monthly_average(df):
    """월평균 계산"""
    try:
        dates = pd.to_datetime(df['effective_date'], errors='coerce').dropna()
        if dates.empty:
            return 0
        months = (dates.max() - dates.min()).days / 30.44
        return round(len(df) / max(1, months), 1)
    except:
        return 0

def get_peak_month(df):
    """피크 월 계산"""
    try:
        df['datetime'] = pd.to_datetime(df['effective_date'], errors='coerce')
        valid_dates = df[df['datetime'].notna()]
        if valid_dates.empty:
            return "N/A"
        peak_month = valid_dates['datetime'].dt.month.mode().iloc[0]
        month_names = ['', '1월', '2월', '3월', '4월', '5월', '6월', 
                      '7월', '8월', '9월', '10월', '11월', '12월']
        return month_names[peak_month]
    except:
        return "N/A"

# 캐시된 크롤러 인스턴스
@st.cache_resource
def get_crawler():
    return FDARealtimeCrawler()

def perform_realtime_update(vectorstore=None, days_back: int = 3) -> Dict[str, Any]:
    """실시간 리콜 데이터 업데이트 수행 - 완전 구현 버전"""
    try:
        # vectorstore가 없으면 기본 경로에서 로드
        if vectorstore is None:
            from utils.chat_recall import recall_vectorstore
            vectorstore = recall_vectorstore
            
        if vectorstore is None:
            return {
                'success': False,
                'error': 'vectorstore not available',
                'message': "벡터스토어에 연결할 수 없습니다"
            }
        
        # 1. 실시간 크롤링 수행
        crawler = get_crawler()
        
        print(f"🔍 최근 {days_back}일간의 리콜 데이터 크롤링 시작...")
        new_recalls = crawler.crawl_latest_recalls(days_back=days_back)
        
        if not new_recalls:
            return {
                'success': True,
                'crawled_count': 0,
                'added_count': 0,
                'message': "새로운 리콜 데이터가 없습니다"
            }
        
        print(f"✅ 크롤링 완료: {len(new_recalls)}건 발견")
        
        # 2. 벡터스토어에 데이터 추가
        print("📥 벡터스토어에 새 데이터 추가 중...")
        added_count = update_vectorstore_with_new_data(new_recalls, vectorstore)
        
        # 3. 결과 반환
        return {
            'success': True,
            'crawled_count': len(new_recalls),
            'added_count': added_count,
            'new_recalls': new_recalls,  # 크롤링된 원본 데이터
            'message': f"새로운 리콜 {len(new_recalls)}건 발견, {added_count}건의 문서 추가됨"
        }
        
    except Exception as e:
        print(f"❌ 실시간 업데이트 오류: {e}")
        return {
            'success': False,
            'error': str(e),
            'crawled_count': 0,
            'added_count': 0,
            'message': f"업데이트 실패: {e}"
        }