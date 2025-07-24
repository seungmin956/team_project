# components/tab_recall.py

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.chat_recall import ask_recall_question, recall_vectorstore
from utils.chat_common_functions import (
    save_chat_history, get_session_keys, initialize_session_state,
    clear_session_state, handle_project_change, display_chat_history,
    update_chat_history, handle_example_question, handle_user_input,
    reset_processing_state
)
from utils.fda_realtime_crawler import create_recall_visualizations
from functools import lru_cache
from datetime import datetime

# 리콜 관련 예시 질문
@lru_cache(maxsize=1)
def get_recall_questions():
    return [
        "지난 달에 새로 발표된 식품 리콜이 있나요?",
        "살모넬라균으로 리콜된 제품 목록을 보여줘.",
        "중금속(납, 카드뮴) 오염 리콜 사례는?",
        "작년 대비 올해 리콜 트렌드에 변화가 있나요?"
    ]

def init_recall_session_state(session_keys):
    """리콜 특화 세션 상태 초기화"""
    initialize_session_state(session_keys)
    
    if "recall_processing_start_time" not in st.session_state:
        st.session_state.recall_processing_start_time = None
    if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
    if "show_charts" not in st.session_state:
        st.session_state.show_charts = False
    if st.session_state.viz_data is None:
        update_visualization_data()


def render_fixed_visualizations():
    """상단 고정 시각화 섹션 - 원인별 차트만 표시"""
    if not st.session_state.show_charts or not st.session_state.viz_data:
        return
    
    # 고정 영역 컨테이너
    viz_container = st.container()
    
    with viz_container:
        st.markdown("""<h1 style="font-size: 20px;"> 리콜 데이터 분석 대시보드</h1>""",unsafe_allow_html=True)
        
        # 통계 요약 카드 (고정 크기)
        stats = st.session_state.viz_data.get('stats', {})
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_recalls = stats.get('total_recalls', 0)
                st.markdown(f"""
                <div style="
                    background-color:#f5f5f5; 
                    padding:20px; 
                    border-radius:10px; 
                    border:1px solid #444;
                    height:140px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    min-width:0;
                ">
                    <p style='font-size:13px;text-align:left;color:#666;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>총 리콜 건수</p>
                    <p style='font-size:25px;text-align:left;font-weight:bold;color:black;margin:8px 0;'>{total_recalls}건</p>
                    <p style='font-size:12px;text-align:left;color:#888;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>전체 벡터DB 문서</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                realtime_count = stats.get('realtime_recalls', 0)
                realtime_ratio = stats.get('realtime_ratio', 0)
                st.markdown(f"""
                <div style="
                    background-color:#f5f5f5; 
                    padding:20px; 
                    border-radius:10px; 
                    border:1px solid #444;
                    height:140px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    min-width:0;
                ">
                    <p style='font-size:13px;text-align:left;color:#666;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>⚡'실시간' 데이터</p>
                    <p style='font-size:25px;text-align:left;font-weight:bold;color:#e74c3c;margin:8px 0;'>{realtime_count}건</p>
                    <p style='font-size:12px;text-align:left;color:#888;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>비율: {realtime_ratio:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                database_count = stats.get('database_recalls', 0)
                st.markdown(f"""
                <div style="
                    background-color:#f5f5f5; 
                    padding:20px; 
                    border-radius:10px; 
                    border:1px solid #444;
                    height:140px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    min-width:0;
                ">
                    <p style='font-size:13px;text-align:left;color:#666;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>📚기존 DB</p>
                    <p style='font-size:25px;text-align:left;font-weight:bold;color:#3498db;margin:8px 0;'>{database_count:,}건</p>
                    <p style='font-size:12px;text-align:left;color:#888;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>사전 구축 데이터</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                latest_crawl = stats.get('latest_crawl', '없음')
                if latest_crawl != '없음' and len(latest_crawl) > 10:
                    display_time = latest_crawl[:10]  # 날짜만
                    display_hour = latest_crawl[11:16]  # 시간만
                else:
                    display_time = latest_crawl
                    display_hour = ""
                
                st.markdown(f"""
                <div style="
                    background-color:#f5f5f5; 
                    padding:20px; 
                    border-radius:10px; 
                    border:1px solid #444;
                    height:140px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    min-width:0;
                ">
                    <p style='font-size:13px;text-align:left;color:#666;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>최근 업데이트</p>
                    <p style='font-size:25px;text-align:left;font-weight:bold;color:#27ae60;margin:4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{display_time}</p>
                    <p style='font-size:12px;text-align:left;color:#888;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{display_hour}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 간격 추가
        st.markdown("<br>", unsafe_allow_html=True)
        
        # st.markdown("---")  # 구분선

def update_visualization_data():
    """시각화 데이터 업데이트"""
    if recall_vectorstore is None:
        return
    
    try:
        viz_data = create_recall_visualizations(recall_vectorstore)
        if viz_data:
            st.session_state.viz_data = viz_data
            st.session_state.show_charts = True
    except Exception as e:
        st.error(f"시각화 데이터 업데이트 오류: {e}")

def render_sidebar_controls(project_name, chat_mode, session_keys):
    """사이드바 컨트롤 패널 렌더링 - 상태 표시만"""
    # 프로젝트 변경 처리
    project_changed = handle_project_change(project_name, chat_mode, session_keys)
    if project_changed:
        st.rerun()
    elif project_name:
        st.success(f"✅ '{project_name}' 진행 중")
    
    # st.markdown("---")
    
    # 기존 버튼들
    has_project_name = bool(project_name and project_name.strip())
    has_chat_history = bool(st.session_state[session_keys["chat_history"]])
    is_processing = st.session_state[session_keys["is_processing"]]
    
    # 저장 버튼
    save_disabled = not (has_project_name and has_chat_history) or is_processing
    if st.button("💾 대화 저장", disabled=save_disabled, use_container_width=True):
        if has_project_name and has_chat_history:
            with st.spinner("저장 중..."):
                success = save_chat_history(
                    project_name.strip(),
                    st.session_state[session_keys["chat_history"]],
                    st.session_state[session_keys["langchain_history"]],
                    chat_mode
                )
                if success:
                    st.success("✅ 저장 완료!")
                else:
                    st.error("❌ 저장 실패")
    
    # 초기화 버튼
    clear_disabled = not (has_project_name and has_chat_history) or is_processing
    if st.button("🗑️ 대화 초기화", disabled=clear_disabled, use_container_width=True):
        clear_session_state(session_keys)
        st.success("초기화 완료")
        st.rerun()
    
    return has_project_name, has_chat_history, is_processing


def render_example_questions(session_keys, is_processing):
    """예시 질문 섹션 렌더링"""
    with st.expander("💡 예시 질문", expanded=False):
        recall_questions = get_recall_questions()
        
        cols = st.columns(2)
        for i, question in enumerate(recall_questions[:4]):
            col_idx = i % 2
            with cols[col_idx]:
                short_question = question[:25] + "..." if len(question) > 25 else question
                
                if st.button(
                    short_question, 
                    key=f"recall_example_{i}", 
                    use_container_width=True, 
                    disabled=is_processing,
                    help=question
                ):
                    handle_example_question(question, session_keys)
                    st.rerun()

def render_chat_area(session_keys, is_processing):
    """메인 채팅 영역 렌더링"""
    # 상단 고정 시각화
    render_fixed_visualizations()
    
    # 예시 질문 섹션
    render_example_questions(session_keys, is_processing)
    
    # 대화 기록 표시
    chat_container = st.container()
    with chat_container:
        display_chat_history(session_keys)
    
    # 질문 처리
    if st.session_state[session_keys["selected_question"]]:
        if not st.session_state.recall_processing_start_time:
            st.session_state.recall_processing_start_time = datetime.now()
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 실시간 데이터 수집 및 분석 중..."):
                try:
                    current_question = st.session_state[session_keys["selected_question"]]
                    
                    # 자동 크롤링이 포함된 질문 처리
                    result = ask_recall_question(
                        current_question, 
                        st.session_state[session_keys["langchain_history"]]
                    )
                    
                    # 챗봇 답변 표시
                    answer = result.get("answer", "답변을 생성할 수 없습니다.")
                    st.markdown(answer)
                    
                    # 처리 시간 표시
                    if st.session_state.recall_processing_start_time:
                        processing_time = (datetime.now() - st.session_state.recall_processing_start_time).total_seconds()
                        st.caption(f"⏱️ 처리 시간: {processing_time:.1f}초")
                    
                    # 실시간 데이터 정보 표시
                    if result.get("has_realtime_data"):
                        st.info(f"⚡ 실시간 데이터 {result.get('realtime_count', 0)}건 포함됨")
                    
                    # 시각화 데이터 업데이트 (고정 영역에 표시됨)
                    update_visualization_data()
                    
                    update_chat_history(
                        current_question, 
                        answer, 
                        session_keys, 
                        result.get("chat_history", [])
                    )
                    
                    reset_processing_state(session_keys)
                    st.session_state.recall_processing_start_time = None
                    
                    # 새 데이터가 추가되었으면 시각화 캐시 클리어
                    if result.get("realtime_count", 0) > 0:
                        st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"답변 생성 중 오류: {str(e)[:100]}...")
                    reset_processing_state(session_keys)
                    st.session_state.recall_processing_start_time = None
                    
                st.rerun()

def show_recall_chat():
    """리콜 전용 챗봇 - 자동 시각화 + 동향 분석 버전"""
    st.info("""
    🔎 **자동 실시간 리콜 분석 시스템** 
    - 질문 시, 최신 리콜 데이터를 실시간으로 자동 수집
    - 기존 DB와 통합하여 리콜 이슈를 분석 제공
    - 저장한 대화는 ‘기획안 요약 도우미’ 탭에서 자동 요약 가능
    """)
    
    chat_mode = "리콜사례"
    session_keys = get_session_keys(chat_mode)
    
    # 세션 상태 초기화
    init_recall_session_state(session_keys)

    # 레이아웃
    col_left, col_center, col_right = st.columns([1, 3, 1])
   
    with col_left:
        # 프로젝트 이름 입력
        project_name = st.text_input(
            "프로젝트 이름", 
            placeholder="리콜 프로젝트명", 
            key="recall_project_input"
        )
        
        # 사이드바 컨트롤 렌더링
        has_project_name, has_chat_history, is_processing = render_sidebar_controls(
            project_name, chat_mode, session_keys
        )

    with col_center:
        # 메인 채팅 영역
        render_chat_area(session_keys, is_processing)
        
        # 사용자 입력
        if not is_processing:
            user_input = st.chat_input(
                "리콜 관련 질문을 입력하세요 (자동으로 최신 데이터 수집 및 분석)", 
                key="recall_chat_input"
            )
            if user_input and user_input.strip():
                if len(user_input.strip()) < 3:
                    st.warning("⚠️ 질문이 너무 짧습니다.")
                else:
                    handle_user_input(user_input.strip(), session_keys)
                    st.rerun()
        else:
            st.info("🔄 실시간 데이터 수집 및 분석 중입니다...")

    with col_right:
        pass