# utils/chat_recall.py - LLM 기반 수치형 질문 처리 통합 버전
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import json
import os
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_teddynote import logging
from utils.fda_realtime_crawler import get_crawler, update_vectorstore_with_new_data
from utils.google_crawler import search_and_extract_news_with_fallback, format_news_for_context

# 🆕 모듈화된 import
from utils.processors.filter_criteria_extractor import FilterCriteriaExtractor
from utils.processors.numerical_processor import NumericalQueryProcessor
from utils.processors.logical_processor import LogicalQueryProcessor
from utils.prompts.recall_prompts import RecallPrompts, TranslationPrompts
from utils.fda_realtime_crawler import get_crawler, update_vectorstore_with_new_data
from utils.google_crawler import search_and_extract_news_with_fallback, format_news_for_context


# LLM 기반 수치형 질문 처리 함수들에 필요한 library import
import re
from collections import Counter

load_dotenv()
logging.langsmith("LLMPROJECT")

class RecallState(TypedDict):
    """리콜 검색 시스템 상태"""
    question: str
    question_en: str  # 영어 번역된 질문
    recall_context: str
    recall_documents: List[Document]
    final_answer: str
    chat_history: List[HumanMessage | AIMessage]
    news_context: str  # 구글 뉴스 컨텍스트 추가
    news_documents: List[Dict]  # 뉴스 문서들 추가
    filter_conditions: Dict[str, Any]  # 🆕 필터링 조건 추가


## ChromaDB 초기화 확인 함수
def verify_chromadb_setup(vectorstore) -> bool:
    """ChromaDB 설정 및 메타데이터 구조 확인"""
    if not vectorstore:
        print("❌ 벡터스토어가 초기화되지 않았습니다.")
        return False
    
    try:
        # 컬렉션 정보 확인
        collection = vectorstore._collection
        doc_count = collection.count()
        print(f"✅ ChromaDB 연결 성공: {doc_count}개 문서")
        
        if doc_count > 0:
            # 샘플 데이터 확인
            sample_data = collection.get(limit=1, include=["metadatas"])
            if sample_data and sample_data.get('metadatas'):
                sample_metadata = sample_data['metadatas'][0]
                print(f"📋 메타데이터 필드: {list(sample_metadata.keys())}")
                
                # ont_ 필드 확인
                ont_fields = [key for key in sample_metadata.keys() if key.startswith('ont_')]
                if ont_fields:
                    print(f"🎯 ont_ 필드 발견: {ont_fields}")
                else:
                    print("⚠️ ont_ 메타데이터 필드가 없습니다. 수치 분석에 제한이 있을 수 있습니다.")
                
                return True
            else:
                print("⚠️ 메타데이터가 없는 문서들입니다.")
                return False
        else:
            print("⚠️ ChromaDB에 문서가 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB 확인 중 오류: {e}")
        return False

def initialize_recall_vectorstore():
    """이 코드는 기존 벡터스토어를 그대로 사용합니다"""
    persist_dir = "./data/chroma_db_recall"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            print("기존 리콜 벡터스토어를 로드합니다...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="FDA_recalls"
            )
            
            collection = vectorstore._collection
            doc_count = collection.count()
            print(f"리콜 벡터스토어 로드 완료 ({doc_count}개 문서)")
            # 🎯 검색 시 최신 데이터 우선, 크롤링으로 최신 데이터 보강
            return vectorstore
                
        except Exception as e:
            print(f"벡터스토어 로드 실패: {e}")
            raise
    else:
        print("❌ 벡터스토어 폴더가 존재하지 않습니다")
        return None

# 전역 벡터스토어 초기화
try:
    recall_vectorstore = initialize_recall_vectorstore()
except Exception as e:
    print(f"벡터스토어 초기화 실패: {e}")
    recall_vectorstore = None

def translation_node(state: RecallState) -> RecallState:
    """조건부 번역 노드 - 고유명사 보존 번역"""
    
    will_use_vectorstore = recall_vectorstore is not None
    
    if will_use_vectorstore:
        # 고유명사 보존 번역 수행
        question_en = translate_with_proper_nouns(state["question"])
        print(f"🔤 고유명사 보존 번역: '{state['question']}' → '{question_en}'")
    else:
        question_en = state["question"]
        print(f"🔤 번역 생략 (웹 검색 전용): '{question_en}'")
    
    # 검색용 키워드 추출
    search_keywords = extract_question_keywords(state["question"])
    
    return {
        **state,
        "question_en": question_en,
        "search_keywords": search_keywords
    }

def translate_with_proper_nouns(korean_text: str) -> str:
    """고유명사를 보존하면서 번역하는 개선된 함수"""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        
        # 🔧 모듈화된 프롬프트 사용
        prompt_template = TranslationPrompts.PROPER_NOUN_TRANSLATION
        prompt = prompt_template.format(korean_text=korean_text)
        
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = response.content.strip()
        
        # 번역 결과 검증 및 후처리
        if translated and len(translated) > 0:
            translated = translated.replace('"', '').replace("'", "")
            if translated.lower().startswith('translation:'):
                translated = translated[12:].strip()
            return translated
        else:
            return korean_text
            
    except Exception as e:
        print(f"고유명사 보존 번역 오류: {e}")
        return korean_text


def is_recall_related_question(question: str) -> bool:
    """질문이 리콜 관련인지 판단하는 함"""
    recall_keywords = [
        "리콜", "회수", "recall", "withdrawal", "safety alert",
        "FDA", "식품안전", "제품 문제", "오염", "contamination",
        "세균", "bacteria", "E.coli", "salmonella", "listeria",
        "알레르기", "allergen", "라벨링", "labeling",
        "식중독", "안전", "위험", "문제", "사고"
    ]
    
    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in recall_keywords)

def recall_search_node(state: RecallState) -> RecallState:
    """벡터DB에서 리콜 관련 문서를 검색하고 실시간 크롤링을 조건부로 수행"""
    
    # 일반 질문이면 검색 생략
    if not is_recall_related_question(state["question"]):
        print(f"일반 질문 감지 - 리콜 검색 생략")
        return {
            **state,
            "recall_context": "",
            "recall_documents": []
        }
    
    if recall_vectorstore is None:
        print("리콜 벡터스토어가 초기화되지 않았습니다.")
        return {
            **state,
            "recall_context": "",
            "recall_documents": []
        }
    
    try:
        # 한 번만 정의 - 실시간 크롤링과 검색 전략 모두에 사
        chat_history = state.get("chat_history", [])
        recent_keywords = ["최근","최신","근래", "recent", "latest", "new", "새로운", "요즘", "현재"]
        is_recent_query = any(keyword in state["question"].lower() for keyword in recent_keywords)
        
        # 세션 내에서 이미 크롤링했는지 확인
        has_crawled_in_session = False
        for msg in chat_history:
            if isinstance(msg, AIMessage) and "⚡실시간:" in msg.content:
                has_crawled_in_session = True
                break
        
        # 실시간 크롤링 조건: 최신 데이터 요청 + 세션 내 미크롤링
        should_crawl = is_recent_query and not has_crawled_in_session
        
        if should_crawl:
            print("🔍 최신 데이터 요청 - 실시간 크롤링 수행")
            try:     
                crawler = get_crawler()
                new_recalls = crawler.crawl_latest_recalls(days_back=7)
                
                if new_recalls:
                    added_count = update_vectorstore_with_new_data(new_recalls, recall_vectorstore)
                    print(f"✅ 새 데이터 {added_count}건 추가됨")
                else:
                    print("📋 새 리콜 데이터 없음")
                    
            except Exception as e:
                print(f"⚠️ 실시간 크롤링 실패: {e}")
        
        # 검색 전략 결정
        if is_recent_query:
            print("📅 최신 데이터 우선 검색 (날짜 정렬 수행)...")
            
            # 전체 데이터 날짜 정렬 로직
            all_data = recall_vectorstore.get()
            all_documents = []

            for i, metadata in enumerate(all_data.get('metadatas', [])):
                if metadata:
                    content = all_data.get('documents', [])[i] if i < len(all_data.get('documents', [])) else ""
                    doc = Document(page_content=content, metadata=metadata)
                    all_documents.append(doc)

            def get_date_for_sorting(doc):
                date_str = doc.metadata.get('effective_date', '1900-01-01')
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    return datetime(1900, 1, 1)

            # 청크 제거 후 날짜순 정렬
            if len(all_documents) > 0 and 'chunk_index' in all_documents[0].metadata:
                url_groups = {}
                for doc in all_documents:
                    url = doc.metadata.get('url', 'unknown')
                    if url not in url_groups or len(doc.page_content) > len(url_groups[url].page_content):
                        url_groups[url] = doc
                unique_recalls = list(url_groups.values())
                unique_recalls.sort(key=get_date_for_sorting, reverse=True)
            else:
                unique_recalls = all_documents
                unique_recalls.sort(key=get_date_for_sorting, reverse=True)

            selected_docs = unique_recalls[:5]
            
        else:
            print("🎯 메타데이터 기반 검색 (유사도 + 필터링)...")
            
            search_query = state.get("question_en", state["question"])
            # 키워드 기반 메타데이터 필터 생성
            question_keywords = extract_question_keywords(state["question"]).lower()
            
            selected_docs = recall_vectorstore.similarity_search(
                search_query, 
                k=10,  # 더 많이 가져와서 후처리
                filter={"document_type": "recall"}
            )
            
            # 메타데이터 기반 후처리 필터링
            filtered_docs = []
            for doc in selected_docs:
                ont_food = doc.metadata.get('ont_food', '').lower()
                if question_keywords in ont_food or ont_food in question_keywords:
                    filtered_docs.append(doc)
            
            # 필터링된 결과가 없으면 원본 결과 사용
            selected_docs = filtered_docs[:5] if filtered_docs else selected_docs[:5]

        # 결과 출력
        print(f"\n🎯 최종 selected_docs:")
        for i, doc in enumerate(selected_docs):
            date = doc.metadata.get('effective_date', 'N/A')
            title = doc.metadata.get('title', '')[:50]
            source = doc.metadata.get('source', '')
            print(f"  {i+1}. {date} | {source} | {title}...")

        context_parts = []
        for doc in selected_docs:
            content_with_meta = f"{doc.page_content}\nSource URL: {doc.metadata.get('url', 'N/A')}"
            context_parts.append(content_with_meta)

        context = "\n\n---\n\n".join(context_parts)
        
        print(f"📊 검색 완료: 총 {len(selected_docs)}건")
        
        return {
            **state,
            "recall_context": context,
            "recall_documents": selected_docs
        }
        
    except Exception as e:
        print(f"검색 오류: {e}")
        return {
            **state,
            "recall_context": "",
            "recall_documents": []
        }

def filtered_search_node(state: RecallState) -> RecallState:
    """필터 조건을 활용한 정확한 ChromaDB 검색 - 디버깅 강화"""
    if recall_vectorstore is None:
        print("리콜 벡터스토어가 초기화되지 않았습니다.")
        return {
            **state,
            "recall_context": "",
            "recall_documents": []
        }
    
    try:
        # 🆕 디버깅 정보 추가
        print(f"🔧 filtered_search_node 시작")
        print(f"🔧 전체 state 키들: {list(state.keys())}")
        
        filter_conditions = state.get("filter_conditions", {})
        print(f"🔧 추출된 필터 조건: {filter_conditions}")
        print(f"🔧 필터 조건 타입: {type(filter_conditions)}")
        print(f"🔧 필터 조건 길이: {len(filter_conditions) if filter_conditions else 0}")
        
        if not filter_conditions:
            print("⚠️ 필터 조건이 없습니다. 기본 검색으로 전환")
            return recall_search_node(state)
        
        # ChromaDB 필터 형식으로 변환
        extractor = FilterCriteriaExtractor(recall_vectorstore)
        chroma_filter = extractor.convert_to_chroma_filter(filter_conditions)
        
        # 문서 타입 필터 추가
        chroma_filter["document_type"] = "recall"
        
        # 필터링된 검색 수행
        search_query = state.get("question_en", state["question"])
        
        print(f"🔍 ChromaDB 필터링 검색: query='{search_query}', filter={chroma_filter}")
        
        # 🆕 OR 조건 처리 개선
        try:
            filtered_docs = recall_vectorstore.similarity_search(
                search_query,
                k=10,
                filter=chroma_filter
            )
        except Exception as search_error:
            print(f"⚠️ 필터링 검색 실패: {search_error}")
            # OR 조건 문제일 수 있으므로 단순화된 필터로 재시도
            simple_filter = {"document_type": "recall"}
            for field, value in filter_conditions.items():
                if not isinstance(value, list):
                    simple_filter[field] = value
            
            print(f"🔄 단순화된 필터로 재시도: {simple_filter}")
            filtered_docs = recall_vectorstore.similarity_search(
                search_query,
                k=10,
                filter=simple_filter
            )
        
        # 나머지 기존 로직...
        if not filtered_docs:
            print("❌ 필터링된 검색 결과 없음. 조건을 완화하여 재검색...")
            # 조건 완화 로직...
        
        selected_docs = filtered_docs[:5]
        
        print(f"✅ 필터링된 검색 완료: {len(selected_docs)}건")
        for i, doc in enumerate(selected_docs):
            date = doc.metadata.get('effective_date', 'N/A')
            title = doc.metadata.get('title', '')[:50]
            food_type = doc.metadata.get('ont_food_type', 'N/A')
            contaminant = doc.metadata.get('ont_contaminant', 'N/A')
            print(f"  {i+1}. {date} | {food_type} | {contaminant} | {title}...")
        
        # 컨텍스트 생성
        context_parts = []
        for doc in selected_docs:
            content_with_meta = f"{doc.page_content}\nSource URL: {doc.metadata.get('url', 'N/A')}"
            context_parts.append(content_with_meta)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            **state,
            "recall_context": context,
            "recall_documents": selected_docs
        }
        
    except Exception as e:
        print(f"필터링된 검색 오류: {e}")
        # 오류 시 기본 검색으로 폴백
        return recall_search_node(state)


def google_news_search_node(state: RecallState) -> RecallState:
    """구글 뉴스에서 리콜 정보를 검색"""
    
    try:
        # 뉴스 검색 전용 키워드 추출
        clean_keywords = extract_question_keywords(state["question"]) # "만두 리콜 사례" → "만두"
        
        print(f"📰 구글 뉴스 검색 시작: '{clean_keywords}' (원본: '{state['question']}')")
        
        # 뉴스 검색 및 본문 추출
        news_results = search_and_extract_news_with_fallback(clean_keywords, max_results=3)
        
        if news_results:
            # 뉴스 컨텍스트 생성
            news_context = format_news_for_context(news_results)
            print(f"✅ 구글 뉴스 검색 완료: {len(news_results)}건")
            
            return {
                **state,
                "recall_context": "",   # 완전 초기화
                "recall_documents": [], # 완전 초기화
                "news_context": news_context,
                "news_documents": news_results
            }
        else:
            print("❌ 관련 뉴스를 찾을 수 없습니다")
            # 뉴스도 없을 때 recall 데이터도 완전 초기화
            return {
                **state,
                "recall_context": "",   # 초기화
                "recall_documents": [], # 초기화
                "news_context": "",
                "news_documents": []
            }
            
    except Exception as e:
        print(f"뉴스 검색 오류: {e}")
        # 오류 시에도 recall 데티터 초기화
        return {
            **state,
            "recall_context": "",   # 초기화
            "recall_documents": [], # 초기화
            "news_context": "",
            "news_documents": []
        }


## 🆕 새로운 지능형 라우터: 필터링 -> 수치형 -> 기본 검색 순서
def enhanced_intelligent_router(state: RecallState) -> str:
    """향상된 지능형 라우터 - 필터 조건 저장 디버깅"""
    
    # 🔧 1순위: 논리형 + 수치형 질문 처리 (필터링보다 먼저!)
    if recall_vectorstore:
        # 논리형 질문 체크 (최우선)
        logical_processor = LogicalQueryProcessor(recall_vectorstore)
        if logical_processor.is_logical_question(state["question"]):
            print("🧠 1순위: LLM이 논리 연산 질문으로 분류 - 논리 분석 수행")
            return "logical_analysis"
        
        # 수치형 질문 체크 (두번째)
        processor = NumericalQueryProcessor(recall_vectorstore)
        if processor.is_numerical_question(state["question"]):
            print("🧠 1순위: LLM이 수치형 질문으로 분류 - 수치 분석 수행")
            return "numerical_analysis"
    
    # 🔧 2순위: 필터링 질문 처리 (논리/수치형 다음으로)
    if recall_vectorstore:
        extractor = FilterCriteriaExtractor(recall_vectorstore)
        filter_result = extractor.extract_filter_conditions(state["question"])
        
        detected_filters = filter_result.get("filters_detected", False)
        confidence = filter_result.get("confidence", 0)
        conditions = filter_result.get("conditions", {})
        
        if detected_filters and confidence > 0.6 and conditions:
            print(f"🎯 2순위: 필터링 질문 감지 (신뢰도: {confidence:.2f}, 조건: {conditions})")
            
            # 🆕 필터 조건 저장 및 디버깅
            state["filter_conditions"] = conditions
            print(f"🔧 state에 저장된 필터 조건: {state.get('filter_conditions', 'NOT_FOUND')}")
            
            return "filtered_search"
        else:
            print(f"⚠️ 필터링 조건 불완전 (신뢰도: {confidence:.2f}, 조건: {conditions})")
    
    # 나머지 기존 로직...
    if not is_recall_related_question(state["question"]):
        print("📝 3순위: 일반 질문 - 답변 생성으로 직행")
        return "generate_answer"
    
    # 기존 검색 결과 기반 판단 로직...
    recall_docs = state.get("recall_documents", [])
    recall_count = len(recall_docs)
    
    print(f"🔍 벡터DB 검색 결과: {recall_count}건")
    
    if recall_count >= 1:
        # 🔧 개선된 매칭 로직 - 더 엄격한 기준 적용
        question_en = state.get("question_en", "").lower()
        question_kr = state["question"].lower()
        
        # 🆕 확장된 불용어 리스트
        english_stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                           'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 
                           'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 
                           'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                           'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                           'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 
                           'should', 'now', 'please', 'tell', 'me', 'case', 'cases']
        korean_stopwords = ['리콜', '사례', '관련', '있나요', '어떤', '무엇', '알려', '주세요', '해주세요']
        
        # 양쪽 언어에서 키워드 추출 및 불용어 제거
        search_terms = []
        if question_en:
            en_terms = [term for term in question_en.split() 
                       if len(term) > 2 and term not in english_stopwords]
            search_terms.extend(en_terms)
        
        kr_terms = [term for term in extract_question_keywords(question_kr).lower().split()
                   if len(term) > 1 and term not in korean_stopwords]
        search_terms.extend(kr_terms)
        
        # 🆕 의미있는 키워드만 추출 (제품명, 브랜드명, 오염물질 등)
        meaningful_terms = [term for term in search_terms if len(term) > 2]
        
        print(f"🔍 정제된 검색 키워드: {meaningful_terms}")
        
        relevant_count = 0
        high_quality_matches = 0  # 🆕 고품질 매칭 카운터
        
        for i, doc in enumerate(recall_docs):
            # 메타데이터 추출
            ont_food = doc.metadata.get('ont_food', '').lower()
            ont_food_type = doc.metadata.get('ont_food_type', '').lower()
            ont_recall_reason = doc.metadata.get('ont_recall_reason', '').lower()
            ont_contaminant = doc.metadata.get('ont_contaminant', '').lower()
            
            # 🆕 메타데이터 품질 검사
            metadata_quality = sum([1 for field in [ont_food, ont_food_type, ont_recall_reason, ont_contaminant] 
                                  if field and field.strip()])
            
            # 제목과 내용에서 키워드 검색
            title = doc.metadata.get('title', '').lower()
            content_preview = doc.page_content[:500].lower()
            
            # 🆕 고품질 매칭 조건: 메타데이터나 제목에서 직접 매칭
            high_quality_matches_found = []
            standard_matches_found = []
            
            for term in meaningful_terms:
                # 고품질 매칭: 메타데이터 직접 매칭
                if (term in ont_food or term in ont_food_type or 
                    term in ont_contaminant or term in title[:100]):
                    high_quality_matches_found.append(term)
                # 일반 매칭: 내용에서 매칭  
                elif term in content_preview:
                    standard_matches_found.append(term)
            
            # 🆕 매칭 품질 평가
            total_matches = high_quality_matches_found + standard_matches_found
            is_high_quality = len(high_quality_matches_found) > 0 and metadata_quality >= 2
            is_standard_relevant = len(total_matches) >= 2  # 최소 2개 키워드 매칭 필요
            
            if is_high_quality:
                relevant_count += 1
                high_quality_matches += 1
                print(f"    ✅ 고품질 매칭 {i+1}: {title[:50]}... (메타데이터 매칭: {high_quality_matches_found})")
                print(f"        메타데이터: food={ont_food}, type={ont_food_type}, 품질점수={metadata_quality}")
            elif is_standard_relevant:
                relevant_count += 1
                print(f"    ✅ 표준 매칭 {i+1}: {title[:50]}... (매칭: {total_matches})")
                print(f"        메타데이터: food={ont_food}, type={ont_food_type}, 품질점수={metadata_quality}")
            else:
                print(f"    ❌ 매칭 실패 {i+1}: {title[:50]}... (매칭: {total_matches}, 품질점수={metadata_quality})")
        
        print(f"🎯 매칭 결과: 고품질={high_quality_matches}건, 전체={relevant_count}/{recall_count}건")
        
        # 🆕 더 엄격한 통과 기준
        if high_quality_matches >= 1 or relevant_count >= 2:
            print("📋 신뢰할 수 있는 관련 문서 발견 - 답변 생성으로 진행")
            return "generate_answer"
        else:
            print("📰 신뢰할 수 있는 관련 문서 없음 - 구글 뉴스 검색 수행")
            return "google_search"
    else:
        print("📰 벡터 검색 결과 없음 - 구글 뉴스 검색 수행")
        return "google_search"


def numerical_analysis_node(state: RecallState) -> RecallState:
    """🔧 모듈화된 수치형 분석 노드"""
    if not recall_vectorstore:
        return {
            **state,
            "final_answer": "벡터스토어가 초기화되지 않아 수치 분석을 수행할 수 없습니다."
        }
    
    try:
        processor = NumericalQueryProcessor(recall_vectorstore)
        result = processor.process_numerical_query(state["question"])
        
        if 'error' in result:
            return {
                **state,
                "final_answer": f"수치 분석 중 오류: {result['error']}"
            }
        
        # LLM으로 답변 생성
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        # 🔧 모듈화된 프롬프트 사용
        prompt = PromptTemplate.from_template(RecallPrompts.NUMERICAL_ANSWER)
        chain = prompt | llm | StrOutputParser()
        
        # 결과 포맷팅
        if result['type'] == 'ranking':
            formatted_result = "\n".join([f"{i+1}. {item['name']}: {item['count']}건" 
                                        for i, item in enumerate(result['result'])])
        elif result['type'] == 'comparison':
            comp_result = result['result']
            formatted_result = f"최근 평균: {comp_result['recent_avg']}건/월, 이전 평균: {comp_result['earlier_avg']}건/월, 추세: {comp_result['trend']}"
        else:
            formatted_result = str(result['result'])
        
        answer = chain.invoke({
            "question": state["question"],
            "analysis_type": result['type'],
            "result": formatted_result,
            "description": result['description']
        })
        
        final_answer = f"{answer}\n\n📊 정보 출처: FDA 공식 데이터베이스 LLM 기반 통계 분석"
        
        return {
            **state,
            "final_answer": final_answer
        }
        
    except Exception as e:
        return {
            **state,
            "final_answer": f"수치형 분석 중 오류: {e}"
        }
    
def logical_analysis_node(state: RecallState) -> RecallState:
    """🔧 LLM 기반 논리형 분석 노드 (완전 재작성)"""
    if not recall_vectorstore:
        return {
            **state,
            "final_answer": "벡터스토어가 초기화되지 않아 논리 분석을 수행할 수 없습니다."
        }
    
    try:
        processor = LogicalQueryProcessor(recall_vectorstore)
        result = processor.process_logical_query(state["question"])
        
        if 'error' in result:
            return {
                **state,
                "final_answer": f"논리 분석 중 오류: {result['error']}"
            }
        
        # LLM으로 답변 생성
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        prompt = PromptTemplate.from_template(RecallPrompts.LOGICAL_ANSWER)
        chain = prompt | llm | StrOutputParser()
        
        # 결과 포맷팅 (LLM 기반 결과에 맞게 수정)
        operation = result.get('operation', '')
        if operation == 'exclude':
            res = result['result']
            formatted_result = f"전체 {res['total_before']}건 → 제외 {res['excluded_count']}건 → 최종 {res['final_count']}건"
        elif operation in ['compare', 'temporal']:
            # 🆕 실제 원인명을 포함한 상세 정보 전달
            comparison_data = {}
            for subject, data in result['result'].items():
                if isinstance(data, dict):
                    comparison_data[subject] = {
                        'total_count': data.get('total_count', 0),
                        'top_reasons': data.get('top_reasons', {}),
                        'top_allergens': data.get('top_allergens', {}),
                        'top_contaminants': data.get('top_contaminants', {}),
                        'top_food_types': data.get('top_food_types', {})
                    }
            
            # JSON 형태로 전달하여 LLM이 파싱할 수 있도록
            formatted_result = json.dumps(comparison_data, ensure_ascii=False, indent=2)
        elif operation == 'conditional':
            res = result['result']
            formatted_result = f"총 {res['total_count']}건"
        else:
            formatted_result = str(result.get('result', ''))
        
        # 관련 링크는 result에서 직접 가져오기
        related_links = "\n".join(result.get('example_links', []))
        if not related_links:
            related_links = "관련 리콜 사례 링크를 추출할 수 없습니다."
        
        answer = chain.invoke({
            "question": state["question"],
            "operation": operation,
            "result": formatted_result,
            "description": result['description'],
            "related_links": related_links
        })
        
        final_answer = f"{answer}\n\n📊 정보 출처: FDA 공식 데이터베이스"
        
        return {
            **state,
            "final_answer": final_answer
        }
        
    except Exception as e:
        return {
            **state,
            "final_answer": f"논리형 분석 중 오류: {e}"
        }


def extract_related_links(result, recall_docs):
    """논리 연산 결과에서 관련 리콜 사례 링크 추출 (개선 버전)"""
    links = []
    
    try:
        # operation 타입에 따라 링크 추출 방식 다르게 처리
        operation = result.get('operation', '')
        
        if operation == 'exclude':
            # 제외 연산: 최종 결과 데이터의 링크
            example_links = result.get('example_links', [])
            if example_links:
                links.extend(example_links[:5])  # 최대 5개
                
        elif operation == 'compare':
            # 비교 연산: 각 비교 대상별 링크
            example_links = result.get('example_links', {})
            for subject, subject_links in example_links.items():
                if subject_links:
                    links.append(f"📊 {subject} 관련:")
                    links.extend(subject_links[:3])  # 각 대상별 최대 3개
                    links.append("")  # 구분용 빈 줄
                    
        elif operation == 'temporal':
            # 시간 비교: 각 기간별 링크
            example_links = result.get('example_links', {})
            for period, period_links in example_links.items():
                if period_links:
                    links.append(f"📅 {period} 관련:")
                    links.extend(period_links[:3])  # 각 기간별 최대 3개
                    links.append("")  # 구분용 빈 줄
                    
        elif operation == 'conditional':
            # 조건부 연산: 최종 결과 데이터의 링크
            example_links = result.get('example_links', [])
            if example_links:
                links.extend(example_links[:5])  # 최대 5개
        
        # 링크가 없으면 기본 문서에서 추출
        if not links:
            for doc in recall_docs[:5]:
                url = doc.metadata.get('url', '')
                title = doc.metadata.get('title', '')
                date = doc.metadata.get('effective_date', '')
                
                if url and title:
                    short_title = title[:50] + "..." if len(title) > 50 else title
                    links.append(f"• {short_title} ({date})\n  {url}")
        
        # 링크가 여전히 없으면 기본 메시지
        if not links:
            return "관련 리콜 사례 링크를 추출할 수 없습니다."
        
        return "\n".join(links)
        
    except Exception as e:
        return f"링크 추출 중 오류: {e}"
    

def _expand_keywords(self, keywords: List[str]) -> List[str]:
    """🆕 더 범용적인 키워드 확장"""
    keyword_mapping = {
        # 세균류
        '살모넬라': ['salmonella'],
        'salmonella': ['살모넬라'],
        '리스테리아': ['listeria'],
        'listeria': ['리스테리아'],
        '대장균': ['e.coli', 'ecoli', 'e coli'],
        'e.coli': ['대장균', 'ecoli'],
        'ecoli': ['대장균', 'e.coli'],
        
        # 식품 카테고리
        '세균': ['bacterial', 'bacteria', 'contamination'],
        'bacterial': ['세균', 'bacteria'],
        '유제품': ['dairy', 'milk', 'cheese', 'yogurt', 'butter', 'cream'],
        'dairy': ['유제품', 'milk', 'cheese'],
        '육류': ['meat', 'beef', 'pork', 'chicken'],
        'meat': ['육류', 'beef', 'pork'],
        '해산물': ['seafood', 'fish', 'salmon', 'shrimp'],
        'seafood': ['해산물', 'fish'],
        
        # 알레르기 관련
        '알레르기': ['allergy', 'allergen', 'allergic'],
        'allergen': ['알레르기', 'allergy'],
        '땅콩': ['peanut', 'groundnut'],
        'peanut': ['땅콩'],
        '견과류': ['nuts', 'tree nuts', 'almonds', 'walnuts'],
        'nuts': ['견과류']
    }
    
    expanded = set()
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        expanded.add(keyword_lower)
        
        if keyword_lower in keyword_mapping:
            expanded.update(keyword_mapping[keyword_lower])
    
    return list(expanded)

def extract_question_keywords(question: str) -> str:
    """사용자 의도를 보존하면서 키워드를 추출"""
    
    # 국가 지시어 확인
    country_indicators = ["국산", "국내", "한국산"]
    is_korean_query = any(indicator in question for indicator in country_indicators)
    
    # 사전 정규화 (하지만 의도는 보존)
    normalized = question
    if is_korean_query:
        # 국산/국내 → 한국으로 통일 (완전 제거하지 않음)
        normalized = normalized.replace("국산", "한국").replace("국내", "한국").replace("한국산", "한국")
    
    # 불용어 제거
    stop_words = ["회수", "사례", "있나요", "관련", "어떤", "최근"]
    
    words = re.findall(r'[가-힣A-Za-z]{2,}', normalized)
    meaningful_words = [word for word in words if word not in stop_words and len(word) >= 2]
    
    # 최대 2개 키워드 (한국 + 제품명)
    result = " ".join(meaningful_words[:2])
    
    print(f"🔍 키워드 추출 (의도 보존): '{question}' → '{result}'")
    return result if result else question

def answer_generation_node(state: RecallState) -> RecallState:
    """검색된 데이터를 바탕으로 적절한 답변을 생성"""
    
    is_recall_question = is_recall_related_question(state["question"])
    
    if not is_recall_question:
        try:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
            # 🔧 모듈화된 프롬프트 사용
            prompt = PromptTemplate.from_template(RecallPrompts.GENERAL_QUESTION)
            chain = prompt | llm | StrOutputParser()
            
            answer = chain.invoke({"question": state["question"]})
            final_answer = f"{answer}\n\n💡 일반 질문으로 처리됨"
            
            return {
                **state,
                "final_answer": final_answer
            }
            
        except Exception as e:
            return {
                **state,
                "final_answer": f"일반 질문 처리 중 오류: {e}"
            }
    
    # 리콜 관련 질문 처리
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        
        recall_context = state.get("recall_context", "").strip()
        news_context = state.get("news_context", "").strip()

        print(f"🔍 recall_context 길이: {len(recall_context)}")
        print(f"🔍 news_context 길이: {len(news_context)}")
        
        # 컨텍스트에 따른 프롬프트 선택
        if news_context:
            print("📰 뉴스 데이터 기반 답변 선택")
            # 🔧 모듈화된 프롬프트 사용
            prompt = PromptTemplate.from_template(RecallPrompts.NEWS_ANSWER)
            context = news_context
            source_type = "최신 뉴스"
        elif recall_context:
            print("📋 FDA 데이터 기반 답변 선택")
            # 🔧 모듈화된 프롬프트 사용
            prompt = PromptTemplate.from_template(RecallPrompts.RECALL_ANSWER)
            context = recall_context
            source_type = "FDA 공식 데이터"
        else:
            return {
                **state,
                "final_answer": "현재 데이터 기준으로 해당 리콜 사례를 확인할 수 없습니다."
            }
        
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "question": state["question"],
            "recall_context": context if recall_context else "",
            "news_context": context if news_context else ""
        })
        
        # 검색 정보 추가
        search_info = f"\n\n📋 정보 출처: {source_type}"
        
        if recall_context:
            recall_docs = state.get("recall_documents", [])
            if recall_docs:
                realtime_count = len([doc for doc in recall_docs 
                                   if doc.metadata.get("source") == "realtime_crawl"])
                search_info += f" (총 {len(recall_docs)}건"
                if realtime_count > 0:
                    search_info += f", ⚡실시간: {realtime_count}건"
                search_info += ")"
        elif news_context:
            news_docs = state.get("news_documents", [])
            search_info += f" (뉴스 {len(news_docs)}건)"
        
        final_answer = f"{answer}{search_info}"
        
        return {
            **state,
            "final_answer": final_answer
        }
        
    except Exception as e:
        return {
            **state,
            "final_answer": f"답변 생성 중 오류: {e}"
        }

def update_history_node(state: RecallState) -> RecallState:
    """채팅 히스토리를 업데이트"""
    try:
        current_history = state.get("chat_history", [])
        
        updated_history = current_history.copy()
        updated_history.append(HumanMessage(content=state["question"]))
        updated_history.append(AIMessage(content=state["final_answer"]))
        
        # 히스토리 길이 제한 (최대 8개 메시지)
        if len(updated_history) > 8:
            updated_history = updated_history[-8:]
        
        return {
            **state,
            "chat_history": updated_history
        }
        
    except Exception as e:
        print(f"히스토리 업데이트 오류: {e}")
        return state

## LLM 기반 워크플로우 설정
def setup_llm_enhanced_workflow():
    """기존 워크플로우에 LLM 기반 수치형 처리 노드 추가 - ChromaDB 전용"""
    
    # ChromaDB 설정 확인
    if recall_vectorstore:
        print("🔧 ChromaDB 설정 확인 중...")
        verify_chromadb_setup(recall_vectorstore)
    
    workflow = StateGraph(RecallState)
    
    # 기존 노드
    workflow.add_node("translate", translation_node)
    workflow.add_node("recall_search", recall_search_node)
    workflow.add_node("google_search", google_news_search_node)
    workflow.add_node("generate_answer", answer_generation_node)
    workflow.add_node("update_history", update_history_node)
    workflow.add_node("numerical_analysis", numerical_analysis_node) # LLM 기반 수치형 분석 노드
    workflow.add_node("logical_analysis", logical_analysis_node)  # 🆕 논리 연산 노드
    workflow.add_node("filtered_search", filtered_search_node)

    # 기존 엣지
    workflow.add_edge(START, "translate")
    workflow.add_edge("translate", "recall_search")
    
    # 조건부 엣지
    workflow.add_conditional_edges("recall_search", enhanced_intelligent_router, {
         "filtered_search": "filtered_search",   
        "logical_analysis": "logical_analysis",
        "numerical_analysis": "numerical_analysis",  # 🆕 LLM 기반 수치 분석 경로 추가
        "google_search": "google_search",
        "generate_answer": "generate_answer"
    })

    workflow.add_edge("filtered_search", "generate_answer")  # 🆕 필터링 검색 → 답변 생성
    workflow.add_edge("google_search", "generate_answer")
    workflow.add_edge("logical_analysis", "update_history") 
    workflow.add_edge("numerical_analysis", "update_history")
    workflow.add_edge("generate_answer", "update_history")
    workflow.add_edge("update_history", END)

    # 그래프 컴파일
    return workflow.compile()

recall_graph = setup_llm_enhanced_workflow() # 🆕 워크플로우 초기화

def ask_recall_question(question: str, chat_history: List = None) -> Dict[str, Any]:
    """리콜 질문을 처리하는 메인 함수"""
    if chat_history is None:
        chat_history = []
    
    try:
        result = recall_graph.invoke({
            "question": question,
            "question_en": "",  # 번역 노드에서 채워짐
            "recall_context": "",
            "recall_documents": [],
            "final_answer": "",
            "chat_history": chat_history,
            "news_context": "",
            "news_documents": [],
            "filter_conditions": {} 
        })
        
        return {
            "answer": result["final_answer"],
            "recall_documents": result["recall_documents"],
            "chat_history": result["chat_history"]
        }
        
    except Exception as e:
        return {
            "answer": f"처리 중 오류가 발생했습니다: {e}",
            "recall_documents": [],
            "chat_history": chat_history
        }