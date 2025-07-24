# utils/google_crawler.py
"""
구글 뉴스 RSS 및 Selenium 기반 리콜 정보 검색 모듈
"""
import feedparser
import time
import re
from datetime import datetime
from typing import List, Dict
from urllib.parse import quote_plus

# Selenium 관련 라이브러리 import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


def search_google_news_rss(keyword: str, max_results: int = 5, days_back: int = 30) -> List[Dict]:
    """이 코드는 구글 뉴스 RSS에서 리콜 관련 뉴스를 검색합니다"""
    try:
        search_strategies = [
            f"{keyword} 리콜",           
            f"{keyword} 미국 리콜",          
            f"{keyword} recall USA",        
            f"{keyword} 제품 회수"           
        ]
        
        all_results = []
        
        for i, search_query in enumerate(search_strategies):
            print(f"🔍 검색 전략 {i+1}: '{search_query}'")
            
            encoded_query = quote_plus(search_query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
            
            feed = feedparser.parse(rss_url)
            print(f"   RSS 결과: {len(feed.entries)}건")
            
            if not feed.entries:
                continue
                
            strategy_results = []
            for entry in feed.entries[:max_results]:
                try:
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    title = entry.title if hasattr(entry, 'title') else '제목 없음'
                    link = entry.link if hasattr(entry, 'link') else ''
                    summary = entry.summary if hasattr(entry, 'summary') else ''
                    source = entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Unknown'
                    
                    is_fda_related = any(term in (title + summary).lower() for term in ['fda', '미국', 'usa', 'america', 'recall', '리콜'])
                    is_recall_relevant = is_recall_related_text(title + " " + summary)
                    
                    if is_fda_related or is_recall_relevant:
                        strategy_results.append({
                            'title': title, 'link': link, 'summary': summary,
                            'published': pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else 'Unknown',
                            'source': source, 'content': '', 'search_strategy': i+1,
                            'is_fda_related': is_fda_related, 'is_recall_related': is_recall_relevant
                        })
                except Exception:
                    continue
            
            all_results.extend(strategy_results)
            if len(all_results) >= max_results:
                break
        
        def sort_priority(item):
            fda_score = 10 if item['is_fda_related'] else 0
            recall_score = 5 if item['is_recall_related'] else 0
            strategy_score = 6 - item['search_strategy']
            return fda_score + recall_score + strategy_score
        
        all_results.sort(key=sort_priority, reverse=True)
        
        unique_results = []
        seen_titles = set()
        for result in all_results:
            if result['title'] not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result['title'])
        
        return unique_results[:max_results]
        
    except Exception:
        return []
    
def search_bing_news_rss(keyword: str, max_results: int = 5) -> List[Dict]:
    """이 코드는 Bing 뉴스 RSS에서 리콜 관련 뉴스를 검색합니다"""
    try:
        search_strategies = [
            f"{keyword} FDA 리콜",           
            f"{keyword} 미국 리콜",          
            f"{keyword} recall USA",        
            f"{keyword} FDA recall"
        ]

        print(f"📝 검색 키워드: '{keyword}'")
        print(f"📝 생성된 전략: {search_strategies}")
        
        all_results = []
        
        for i, search_query in enumerate(search_strategies):
            print(f"🔍 Bing 검색 전략 {i+1}: '{search_query}'")
            
            encoded_query = quote_plus(search_query)
            rss_url = f"https://www.bing.com/news/search?q={encoded_query}&format=RSS"
            
            # RSS 파싱
            feed = feedparser.parse(rss_url)
            print(f"   Bing RSS 결과: {len(feed.entries)}건")
            
            if not feed.entries:
                continue
                
            strategy_results = []
            for entry in feed.entries[:max_results]:
                try:
                    # Bing RSS 구조에 맞게 파싱
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    title = entry.title if hasattr(entry, 'title') else '제목 없음'
                    link = entry.link if hasattr(entry, 'link') else ''
                    summary = entry.summary if hasattr(entry, 'summary') else ''
                    
                    # Bing은 source 구조가 다름
                    source = "Bing News"
                    
                    is_fda_related = any(term in (title + summary).lower() 
                                       for term in ['fda', '미국', 'usa', 'america', 'recall', '리콜'])
                    is_recall_relevant = is_recall_related_text(title + " " + summary)
                    
                    if is_fda_related or is_recall_relevant:
                        strategy_results.append({
                            'title': title,
                            'link': link, 
                            'summary': summary,
                            'published': pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else 'Unknown',
                            'source': source,
                            'content': '',
                            'search_strategy': i+1,
                            'is_fda_related': is_fda_related,
                            'is_recall_related': is_recall_relevant,
                            'search_engine': 'bing'  # 구분용
                        })
                except Exception:
                    continue
            
            all_results.extend(strategy_results)
            if len(all_results) >= max_results:
                break
        
        # 우선순위 정렬 (기존과 동일)
        def sort_priority(item):
            fda_score = 10 if item['is_fda_related'] else 0
            recall_score = 5 if item['is_recall_related'] else 0
            strategy_score = 6 - item['search_strategy']
            return fda_score + recall_score + strategy_score
        
        all_results.sort(key=sort_priority, reverse=True)
        
        # 중복 제거
        unique_results = []
        seen_titles = set()
        for result in all_results:
            if result['title'] not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result['title'])
        
        return unique_results[:max_results]
        
    except Exception as e:
        print(f"Bing RSS 검색 오류: {e}")
        return []
    
def search_and_extract_news_with_fallback(keyword: str, max_results: int = 3) -> List[Dict]:
    """이 코드는 단순하고 정확한 단일 검색을 수행합니다 (bing 검색+ google 백업 크롤링)"""
    
    # 🆕 한국 제품인 경우 핵심 제품명만 추출
    korean_indicators = ["한국", "국산", "국내", "Korean"]
    
    final_keyword = keyword
    for indicator in korean_indicators:
        final_keyword = final_keyword.replace(indicator, "").strip()
    
    # 🆕 최종 검색 키워드 결정
    search_keyword = final_keyword if (final_keyword and len(final_keyword) > 1) else keyword
    
    print(f"🔍 단일 검색: '{search_keyword}' (원본: '{keyword}')")
    
    # 🆕 Bing 우선, Google 백업 (기존 fallback 로직 유지)
    all_results = []
    
    # 1순위: Bing RSS 시도
    try:
        print("🔍 Bing 뉴스 RSS 검색 시도...")
        bing_results = search_bing_news_rss(search_keyword, max_results)
        
        if bing_results:
            print(f"✅ Bing 검색 성공: {len(bing_results)}건")
            all_results.extend(bing_results)
        else:
            print("❌ Bing에서 관련 뉴스를 찾을 수 없음")
            
    except Exception as e:
        print(f"❌ Bing 검색 실패: {e}")
    
    # 2순위: 결과가 없으면 Google 백업
    if not all_results:
        try:
            print("🔄 Google RSS 백업 검색 시도...")
            google_results = search_google_news_rss(search_keyword, max_results)
            
            if google_results:
                # 검색 엔진 구분을 위해 메타데이터 추가
                for result in google_results:
                    result['search_engine'] = 'google'
                
                all_results.extend(google_results)
                print(f"✅ Google 백업 검색 성공: {len(google_results)}건")
                
        except Exception as e:
            print(f"❌ Google 백업 검색도 실패: {e}")
    
    # 🆕 결과 확인
    if not all_results:
        print("❌ 모든 검색 엔진에서 관련 뉴스를 찾을 수 없습니다")
        return []
    
    # 본문 추출 (기존 로직과 동일)
    enriched_results = []
    for i, news_item in enumerate(all_results):
        engine = news_item.get('search_engine', 'bing')
        print(f"({i+1}/{len(all_results)}) [{engine}] 본문 추출 중: {news_item['title'][:50]}...")
        
        content = extract_news_content_selenium(news_item['link'])
        
        if content and len(content) > 50:
            news_item['content'] = content
        else:
            summary_text = BeautifulSoup(news_item.get('summary', ''), 'html.parser').get_text()
            news_item['content'] = summary_text
        
        enriched_results.append(news_item)
        time.sleep(1)  # 서버 부하 방지
    
    print(f"✅ 최종 검색 완료: {len(enriched_results)}건")
    return enriched_results


def is_recall_related_text(text: str) -> bool:
    """이 코드는 텍스트가 리콜 관련인지 확인합니다"""
    recall_keywords = [
        "리콜", "회수", "recall", "withdrawal", "식품안전", "오염", "contamination",
        "세균", "bacteria", "안전경고", "위험", "식중독", "알레르기", "라벨링",
        "문제", "결함", "하자", "부적합", "판매중단", "유통중단", "반품"
    ]
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in recall_keywords)


# 전역 드라이버 캐시 (세션당 재사용)
_driver_cache = None

def get_optimized_driver():
    """이 코드는 최적화된 Selenium 드라이버를 재사용 가능하게 제공합니다"""
    global _driver_cache
    
    if _driver_cache is None:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")  # 이미지 로딩 비활성화
        options.add_argument("--disable-javascript")  # JS 비활성화 (뉴스 본문은 대부분 정적)
        options.add_argument("--page-load-strategy=eager")  # DOM 로딩만 기다림
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        _driver_cache = webdriver.Chrome(service=service, options=options)
        _driver_cache.set_page_load_timeout(10)  # 타임아웃 단축
    
    return _driver_cache

def close_driver_cache():
    """이 코드는 캐시된 드라이버를 정리합니다"""
    global _driver_cache
    if _driver_cache:
        try:
            _driver_cache.quit()
        except:
            pass
        finally:
            _driver_cache = None

def extract_news_content_selenium(url: str) -> str:
    """이 코드는 최적화된 Selenium으로 뉴스 본문을 추출합니다"""
    try:
        driver = get_optimized_driver()
        
        # 페이지 접속 (타임아웃 단축)
        driver.get(url)
        time.sleep(1)  # 대기 시간 단축 (3초 → 1초)

        # 페이지 소스를 BeautifulSoup으로 파싱
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 불필요한 태그 제거
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # 본문 추출 선택자 (효율적인 순서로 재배열)
        content_selectors = [
            'article',  # 가장 일반적
            '.article-content', '.news-content', '.post-content', 
            'div[class*="content"]', 'div[class*="article"]'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                best_text = max((elem.get_text().strip() for elem in elements), 
                              key=len, default="")
                if len(best_text) > len(content):
                    content = best_text
                if len(content) > 300:  # 충분한 내용이면 중단
                    break
        
        # p 태그 백업 추출
        if len(content) < 100:
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') 
                         if len(p.get_text().strip()) > 30]
            content = '\n'.join(paragraphs)
        
        # 후처리
        if content:
            content = re.sub(r'\s+', ' ', content).strip()[:2000]  # 길이 제한 단축
        
        print(f"  ✅ 최적화 크롤링 성공: {len(content)}자")
        return content
        
    except Exception as e:
        print(f"  ❌ 크롤링 실패: {url} - {e}")
        return ""


def format_news_for_context(news_results: List[Dict]) -> str:
    """이 코드는 뉴스 결과를 컨텍스트 형태로 포맷합니다"""
    if not news_results:
        return ""
    
    context_parts = []
    for news in news_results:
        news_context = f"""
뉴스 제목: {news.get('title', '')}
출처: {news.get('source', 'Unknown')}
발행일: {news.get('published', 'Unknown')}
URL: {news.get('link', '')}

기사 내용:
{news.get('content', '')}
        """.strip()
        context_parts.append(news_context)
    
    return "\n\n---\n\n".join(context_parts)
