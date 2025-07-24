# utils/processors/logical_processor.py - 개선된 버전

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class LogicalQueryProcessor:
    """LLM 기반 논리 연산(제외, 비교, 조건부) 질문 처리 클래스 - 개선된 버전"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.filter_cache = {}
        self.exclude_cache = {}
        self.metadata_sample = None
        # 🆕 현재 날짜 정보 추가
        self.current_date = datetime.now()
        self.current_year = self.current_date.year
    
    def _resolve_relative_time(self, time_expression: str) -> str:
        """상대적 시간 표현을 절대 연도로 변환"""
        time_lower = time_expression.lower().strip()
        
        # 상대적 시간 매핑
        if time_lower in ['올해', '이번년', '이번 년', '현재년', '2025년']:
            return str(self.current_year)  # 2025
        elif time_lower in ['작년', '지난해', '지난 해', '작년도']:
            return str(self.current_year - 1)  # 2024
        elif time_lower in ['재작년', '재작년도']:
            return str(self.current_year - 2)  # 2023
        
        # 절대 연도 추출
        year_match = re.search(r'(\d{4})', time_expression)
        if year_match:
            return year_match.group(1)
        
        # 매핑 실패 시 원본 반환
        return time_expression
    
    def analyze_logical_intent(self, question: str) -> Dict[str, Any]:
        """LLM을 활용한 논리 연산 의도 분석 - 현재 날짜 정보 포함"""
        
        # 🆕 현재 날짜 정보를 프롬프트에 포함
        current_date_str = self.current_date.strftime("%Y년 %m월 %d일")
        
        prompt = f"""
오늘 날짜: {current_date_str} (따라서 올해={self.current_year}년, 작년={self.current_year-1}년)

다음 질문을 분석하여 JSON 형태로 정확히 응답하세요. 반드시 JSON만 반환하고 다른 설명은 하지 마세요.

질문: "{question}"

분석 기준:
1. query_type:
   - "logical": 논리 연산이 필요한 질문 (제외, 비교, 조건부 등)
   - "semantic": 일반적인 검색 질문
   - "numerical": 개수, 순위 등 수치형 질문

2. operation (logical인 경우만):
   - "exclude": 특정 조건을 제외 ("~를 제외한", "~빼고", "~외에")
   - "compare": 두 대상의 비교 ("~와 비교", "~대비", "차이점", "간 비교", "중 어느 쪽", "더 많아")
   - "temporal": 시간별 비교 ("작년과 올해", "월별", "기간별", "년도별", "원인 비교")
   - "conditional": 복합 조건 ("~이면서 ~이 아닌", "~인 경우에만", "~일 때")

3. main_subjects: 주요 분석 대상들 (배열)
4. exclude_conditions: 제외할 조건들 (배열) 
5. conditional_filters: 복합 조건들 (배열)
6. time_periods: 시간 기간들 (배열) - 상대적 표현도 포함
7. confidence: 분석 확신도 (0.0-1.0)

중요한 시간 매핑:
- "올해" → {self.current_year}년
- "작년" → {self.current_year-1}년
- "재작년" → {self.current_year-2}년

예시:
- "작년과 올해의 리콜 원인에 대해서 비교해줘"
  → {{"query_type": "logical", "operation": "temporal", "time_periods": ["작년", "올해"], "main_subjects": ["리콜 원인"], "confidence": 0.95}}

JSON 결과:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # JSON 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            intent = json.loads(result_text)
            
            # 기본값 설정
            intent.setdefault('query_type', 'semantic')
            intent.setdefault('confidence', 0.5)
            intent.setdefault('main_subjects', [])
            intent.setdefault('exclude_conditions', [])
            intent.setdefault('conditional_filters', [])
            intent.setdefault('time_periods', [])
            
            print(f"🧠 논리 연산 의도 분석: {intent}")
            return intent
            
        except Exception as e:
            print(f"⚠️ 논리 의도 분석 실패: {e}")
            return {
                'query_type': 'semantic',
                'confidence': 0.5,
                'main_subjects': [],
                'exclude_conditions': [],
                'conditional_filters': [],
                'time_periods': []
            }
    
    def _filter_by_llm_understanding(self, metadatas: List[Dict], documents: List[str], filter_description: str) -> List[Tuple[Dict, str]]:
        """LLM이 메타데이터를 직접 이해해서 필터링 - 현재 날짜 정보 포함"""
        
        cache_key = filter_description.lower().strip()
        if cache_key in self.filter_cache:
            filter_logic = self.filter_cache[cache_key]
            print(f"🔄 캐시된 필터링 로직 사용: {filter_description}")
        else:
            sample_metadata = self._get_metadata_sample()
            actual_values = self._get_actual_metadata_values(metadatas[:50])
            
            # 🆕 현재 날짜 정보를 프롬프트에 포함
            current_date_str = self.current_date.strftime("%Y년 %m월 %d일")
            
            prompt = f"""
오늘 날짜: {current_date_str} (올해={self.current_year}년, 작년={self.current_year-1}년)

당신은 FDA 리콜 데이터 필터링 전문가입니다.
다음 조건에 맞는 데이터를 찾기 위한 필터링 로직을 JSON으로 작성하세요.

필터링 조건: "{filter_description}"

메타데이터 구조 예시:
{json.dumps(sample_metadata, indent=2)}

실제 데이터에서 발견되는 값들:
{json.dumps(actual_values, indent=2)}

다음 형식으로 응답하세요:
{{
    "filter_logic": [
        {{
            "field": "메타데이터 필드명",
            "condition": "contains|equals|exists|not_exists|starts_with",
            "value": "찾을 값 (exists/not_exists면 null)",
            "reasoning": "이 조건을 설정한 이유"
        }}
    ],
    "combination": "AND|OR"
}}

중요한 매핑 규칙 (현재 날짜 기준):
- "올해" → effective_date가 "{self.current_year}"로 시작
- "작년" → effective_date가 "{self.current_year-1}"로 시작  
- "재작년" → effective_date가 "{self.current_year-2}"로 시작
- "육류" → ont_food_type에서 "Livestock Products" 찾기
- "유제품" → ont_food_type에서 "Dairy Products" 찾기
- "해산물" → ont_food_type에서 "Seafood Products" 찾기
- "알레르기 관련" → ont_allergen 필드에 값이 존재하는지 확인

실제 데이터를 참고하여 정확한 값을 사용하세요.

JSON만 반환:"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                result_text = response.content.strip()
                
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].strip()
                
                filter_logic = json.loads(result_text)
                self.filter_cache[cache_key] = filter_logic
                print(f"🆕 새 필터링 로직 생성: {filter_description}")
                print(f"📋 생성된 로직: {filter_logic}")
                
            except Exception as e:
                print(f"⚠️ LLM 필터링 로직 생성 실패: {e}")
                return []
        
        return self._apply_llm_filter_logic(metadatas, documents, filter_logic)
    
    def _handle_conditional_query_llm(self, intent: Dict, metadatas: List[Dict], documents: List[str]) -> Dict[str, Any]:
        """LLM 기반 조건부 처리 - 복합 조건 개선"""
        conditional_filters = intent.get('conditional_filters', [])
        main_subjects = intent.get('main_subjects', [])
        
        # conditional_filters가 없으면 main_subjects 사용
        if not conditional_filters and main_subjects:
            conditional_filters = main_subjects
        
        if not conditional_filters:
            return {'error': '조건부 필터링할 대상이 없습니다.'}
        
        print(f"🔍 LLM 조건부 필터링: {conditional_filters}")
        
        # 🆕 복합 조건 파싱 ("유제품이면서 알레르기 관련이 아닌")
        positive_conditions = []
        negative_conditions = []
        
        for condition in conditional_filters:
            if '아닌' in condition or '없는' in condition or '제외' in condition:
                # 부정 조건
                clean_condition = condition.replace('이 아닌', '').replace('가 아닌', '').replace('관련이 아닌', '관련').replace('제외', '').strip()
                negative_conditions.append(clean_condition)
            else:
                # 긍정 조건
                positive_conditions.append(condition)
        
        print(f"✅ 긍정 조건: {positive_conditions}")
        print(f"❌ 부정 조건: {negative_conditions}")
        
        # 1단계: 긍정 조건 필터링
        if positive_conditions:
            positive_description = ', '.join(positive_conditions)
            filtered_data = self._filter_by_llm_understanding(metadatas, documents, positive_description)
        else:
            filtered_data = list(zip(metadatas, documents))
        
        print(f"1단계 긍정 조건 필터링 후: {len(filtered_data)}건")
        
        # 2단계: 부정 조건 제외
        if negative_conditions:
            final_data = []
            excluded_data = []
            
            for metadata, doc in filtered_data:
                should_exclude = False
                for neg_condition in negative_conditions:
                    if self._matches_negative_condition(metadata, neg_condition):
                        should_exclude = True
                        break
                
                if should_exclude:
                    excluded_data.append((metadata, doc))
                else:
                    final_data.append((metadata, doc))
            
            print(f"2단계 부정 조건 제외 후: {len(final_data)}건 (제외: {len(excluded_data)}건)")
        else:
            final_data = filtered_data
            excluded_data = []
        
        # 결과 분석
        condition_analysis = {}
        example_links = []
        
        # 리콜 원인별 분석
        reasons = []
        allergens = []
        contaminants = []
        food_types = []
        
        for metadata, doc in final_data:
            food_type = metadata.get('ont_food_type', '기타')
            reason = metadata.get('ont_recall_reason', '기타')
            allergen = metadata.get('ont_allergen', '')
            contaminant = metadata.get('ont_contaminant', '')
            
            reasons.append(reason)
            food_types.append(food_type)
            if allergen and allergen.lower() not in ['null', 'none', '']:
                allergens.append(allergen)
            if contaminant and contaminant.lower() not in ['null', 'none', '']:
                contaminants.append(contaminant)
            
            if food_type not in condition_analysis:
                condition_analysis[food_type] = {
                    'count': 0,
                    'reasons': []
                }
            
            condition_analysis[food_type]['count'] += 1
            condition_analysis[food_type]['reasons'].append(reason)
        
        # 예시 링크 생성
        for metadata, doc in final_data[:5]:
            url = metadata.get('url', '')
            title = metadata.get('title', '')
            date = metadata.get('effective_date', '')
            if url and title:
                short_title = title[:50] + "..." if len(title) > 50 else title
                example_links.append(f"• {short_title} ({date})\n  {url}")
        
        # 조건별 상위 이유 정리
        for food_type in condition_analysis:
            reasons_list = condition_analysis[food_type]['reasons']
            condition_analysis[food_type]['top_reasons'] = dict(Counter(reasons_list).most_common(3))
            del condition_analysis[food_type]['reasons']
        
        # 전체 통계
        reason_counts = Counter(reasons)
        allergen_counts = Counter(allergens)
        contaminant_counts = Counter(contaminants)
        food_type_counts = Counter(food_types)
        
        return {
            'type': 'conditional',
            'operation': 'conditional',
            'result': {
                'total_count': len(final_data),
                'excluded_count': len(excluded_data),
                'condition_analysis': condition_analysis,
                'positive_conditions': positive_conditions,
                'negative_conditions': negative_conditions,
                'top_reasons': dict(reason_counts.most_common(5)),
                'top_allergens': dict(allergen_counts.most_common(3)),
                'top_contaminants': dict(contaminant_counts.most_common(3)),
                'top_food_types': dict(food_type_counts.most_common(3))
            },
            'description': f"{', '.join(conditional_filters)} 조건에 맞는 {len(final_data)}건의 리콜 사례",
            'example_links': example_links,
            'intent': intent
        }

    def _matches_negative_condition(self, metadata: Dict, condition: str) -> bool:
        """부정 조건 매칭 확인"""
        condition_lower = condition.lower()
        
        # 알레르기 관련 체크
        if '알레르기' in condition_lower:
            allergen_value = metadata.get('ont_allergen')
            return allergen_value is not None and str(allergen_value).lower() not in ['null', 'none', '']
        
        # 기타 조건들은 일반 매칭
        searchable_fields = [
            'ont_contaminant', 'ont_allergen', 'ont_recall_reason',
            'ont_food', 'ont_food_type', 'title'
        ]
        
        for field in searchable_fields:
            field_value = str(metadata.get(field, '')).lower()
            if condition_lower in field_value:
                return True
        
        return False

    def _handle_exclude_query_llm(self, intent: Dict, metadatas: List[Dict], documents: List[str]) -> Dict[str, Any]:
        """LLM 기반 제외 연산 처리"""
        main_subjects = intent.get('main_subjects', [])
        exclude_conditions = intent.get('exclude_conditions', [])
        
        print(f"🔍 LLM 제외 연산: {main_subjects}에서 {exclude_conditions} 제외")
        
        # 1단계: 주 대상 필터링
        if main_subjects:
            main_description = ', '.join(main_subjects)
            filtered_data = self._filter_by_llm_understanding(metadatas, documents, main_description)
        else:
            filtered_data = list(zip(metadatas, documents))
        
        print(f"1단계 필터링 후: {len(filtered_data)}건")
        
        # 2단계: 제외 조건 적용
        if exclude_conditions:
            final_data = []
            excluded_data = []
            
            print(f"🚫 제외 조건 적용 중: {exclude_conditions}")
            
            for metadata, doc in filtered_data:
                if self._llm_should_exclude(metadata, exclude_conditions):
                    excluded_data.append((metadata, doc))
                else:
                    final_data.append((metadata, doc))
            
            print(f"✅ 제외 처리 완료: 제외 {len(excluded_data)}건, 최종 {len(final_data)}건")
            
        else:
            final_data = filtered_data
            excluded_data = []
        
        return self._format_exclude_results(filtered_data, final_data, excluded_data, intent)

    def _llm_should_exclude(self, metadata: Dict, exclude_conditions: List[str]) -> bool:
        """LLM이 개별 문서의 제외 여부 판단"""
        
        # 캐싱을 위한 키 생성
        cache_key = f"{metadata.get('url', '')}-{'-'.join(exclude_conditions)}"
        if cache_key in self.exclude_cache:
            return self.exclude_cache[cache_key]
        
        prompt = f"""
    다음 리콜 데이터가 제외 조건에 해당하는지 판단하세요.

    제외 조건: {', '.join(exclude_conditions)}

    리콜 데이터:
    - 식품 종류: {metadata.get('ont_food_type', 'N/A')}
    - 식품명: {metadata.get('ont_food', 'N/A')}
    - 리콜 사유: {metadata.get('ont_recall_reason', 'N/A')}
    - 알레르기 요소: {metadata.get('ont_allergen', 'N/A')}
    - 오염 물질: {metadata.get('ont_contaminant', 'N/A')}
    - 제목: {metadata.get('title', 'N/A')[:100]}

    이 데이터가 제외 조건에 해당하면 "YES", 아니면 "NO"로만 답하세요.

    형식: YES 또는 NO"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            
            should_exclude = result.startswith('YES')
            self.exclude_cache[cache_key] = should_exclude
            
            return should_exclude
            
        except Exception as e:
            print(f"⚠️ LLM 제외 판단 실패: {e}")
            return False

    def _format_exclude_results(self, original_data: List, final_data: List, excluded_data: List, intent: Dict) -> Dict[str, Any]:
        """제외 연산 결과 포맷팅"""
        excluded_count = len(excluded_data)
        
        # 제외된 사례의 주요 원인 분석
        excluded_reasons = []
        for metadata, doc in excluded_data:
            reason = metadata.get('ont_recall_reason', '기타')
            allergen = metadata.get('ont_allergen', '')
            if allergen:
                reason += f" ({allergen})"
            excluded_reasons.append(reason)
        
        excluded_reason_counts = Counter(excluded_reasons)
        
        # 최종 결과의 주요 원인 분석
        final_reasons = []
        example_links = []
        for metadata, doc in final_data[:5]:
            reason = metadata.get('ont_recall_reason', '기타')
            contaminant = metadata.get('ont_contaminant', '')
            if contaminant:
                reason += f" ({contaminant})"
            final_reasons.append(reason)
            
            # 예시 링크 추가
            url = metadata.get('url', '')
            title = metadata.get('title', '')
            date = metadata.get('effective_date', '')
            if url and title:
                short_title = title[:50] + "..." if len(title) > 50 else title
                example_links.append(f"• {short_title} ({date})\n  {url}")
        
        final_reason_counts = Counter(final_reasons)
        
        main_subjects = intent.get('main_subjects', [])
        exclude_conditions = intent.get('exclude_conditions', [])
        
        return {
            'type': 'exclude',
            'operation': 'exclude',
            'result': {
                'total_before': len(original_data),
                'excluded_count': excluded_count,
                'final_count': len(final_data),
                'excluded_reasons': dict(excluded_reason_counts.most_common(5)),
                'final_reasons': dict(final_reason_counts.most_common(5))
            },
            'description': f"{', '.join(main_subjects)} 중 {', '.join(exclude_conditions)}를 제외한 {len(final_data)}건의 리콜 사례",
            'example_links': example_links,
            'intent': intent
        }

    
    def _handle_compare_query_llm(self, intent: Dict, metadatas: List[Dict], documents: List[str]) -> Dict[str, Any]:
        """LLM 기반 비교 연산 처리 - 원인 분석 강화"""
        subjects = intent.get('main_subjects', [])
        time_periods = intent.get('time_periods', [])
        
        # 시간 비교인 경우
        if time_periods and len(time_periods) >= 2:
            # 🆕 상대적 시간을 절대 연도로 변환
            resolved_periods = []
            for period in time_periods:
                resolved_year = self._resolve_relative_time(period)
                resolved_periods.append(resolved_year)
            
            subjects = resolved_periods
            is_temporal = True
            print(f"🕐 시간 매핑: {time_periods} → {resolved_periods}")
        else:
            is_temporal = False
        
        if len(subjects) < 2:
            return {'error': '비교할 대상이 2개 이상 필요합니다.'}
        
        print(f"🔍 LLM {'시간' if is_temporal else '일반'} 비교 연산: {subjects} 간 비교")
        
        comparison_results = {}
        example_links = {}
        
        for subject in subjects:
            if is_temporal:
                # 🆕 연도 기반 필터링
                filtered_data = []
                year = subject  # 이미 resolved된 연도
                for metadata, doc in zip(metadatas, documents):
                    effective_date = metadata.get('effective_date', '')
                    if effective_date.startswith(year):
                        filtered_data.append((metadata, doc))
            else:
                filtered_data = self._filter_by_llm_understanding(metadatas, documents, subject)
            
            print(f"📊 {subject}: {len(filtered_data)}건 데이터")
            
            # 🆕 리콜 원인 상세 분석
            reasons = []
            allergens = []
            contaminants = []
            food_types = []
            links = []
            monthly_counts = defaultdict(int)
            
            for metadata, doc in filtered_data:
                # 다양한 원인 정보 수집
                reason = metadata.get('ont_recall_reason', '기타')
                allergen = metadata.get('ont_allergen', '')
                contaminant = metadata.get('ont_contaminant', '')
                food_type = metadata.get('ont_food_type', '기타')
                
                reasons.append(reason)
                if allergen and allergen.lower() not in ['null', 'none', '']:
                    allergens.append(allergen)
                if contaminant and contaminant.lower() not in ['null', 'none', '']:
                    contaminants.append(contaminant)
                food_types.append(food_type)
                
                # 시간 비교용 월별 분포
                if is_temporal:
                    date_str = metadata.get('effective_date', '')
                    if date_str:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            month_key = date_obj.strftime('%Y-%m')
                            monthly_counts[month_key] += 1
                        except:
                            pass
            
            # 예시 링크 (상위 3개)
            for metadata, doc in filtered_data[:3]:
                url = metadata.get('url', '')
                title = metadata.get('title', '')
                date = metadata.get('effective_date', '')
                if url and title:
                    short_title = title[:50] + "..." if len(title) > 50 else title
                    links.append(f"• {short_title} ({date})\n  {url}")
            
            # 🆕 상세 원인 분석
            reason_counts = Counter(reasons)
            allergen_counts = Counter(allergens)
            contaminant_counts = Counter(contaminants)
            food_type_counts = Counter(food_types)
            
            result_data = {
                'total_count': len(filtered_data),
                'top_reasons': dict(reason_counts.most_common(5)),  # 상위 5개로 확장
                'top_allergens': dict(allergen_counts.most_common(3)),
                'top_contaminants': dict(contaminant_counts.most_common(3)),
                'top_food_types': dict(food_type_counts.most_common(3)),
                'avg_monthly': len(filtered_data) / 12 if len(filtered_data) > 0 else 0
            }
            
            # 시간 비교인 경우 월별 분포 추가
            if is_temporal:
                result_data['monthly_distribution'] = dict(monthly_counts)
            
            comparison_results[subject] = result_data
            example_links[subject] = links
        
        # 🆕 시간 비교인 경우 원인별 변화 분석
        cause_analysis = {}
        if is_temporal and len(comparison_results) >= 2:
            periods = list(comparison_results.keys())
            recent_data = comparison_results[periods[-1]]
            previous_data = comparison_results[periods[0]]
            
            # 전체 건수 변화
            recent_count = recent_data['total_count']
            previous_count = previous_data['total_count']
            
            if previous_count > 0:
                change_rate = ((recent_count - previous_count) / previous_count) * 100
                trend = "증가" if change_rate > 10 else "감소" if change_rate < -10 else "유지"
            else:
                change_rate = 0
                trend = "데이터 부족"
            
            # 주요 원인별 변화 분석
            cause_changes = {}
            recent_reasons = recent_data.get('top_reasons', {})
            previous_reasons = previous_data.get('top_reasons', {})
            
            all_reasons = set(list(recent_reasons.keys()) + list(previous_reasons.keys()))
            for reason in all_reasons:
                recent_cnt = recent_reasons.get(reason, 0)
                previous_cnt = previous_reasons.get(reason, 0)
                
                if previous_cnt > 0:
                    reason_change = ((recent_cnt - previous_cnt) / previous_cnt) * 100
                else:
                    reason_change = 100 if recent_cnt > 0 else 0
                
                cause_changes[reason] = {
                    'previous': previous_cnt,
                    'recent': recent_cnt,
                    'change_rate': round(reason_change, 1)
                }
            
            cause_analysis = {
                'total_change_rate': round(change_rate, 1),
                'trend': trend,
                'cause_changes': cause_changes
            }
        
        result = {
            'type': 'temporal' if is_temporal else 'compare',
            'operation': 'temporal' if is_temporal else 'compare',
            'result': comparison_results,
            'description': f"{', '.join(subjects)} 간 리콜 사례 비교 분석 (원인별 상세 분석 포함)",
            'example_links': example_links,
            'intent': intent
        }
        
        if cause_analysis:
            result['cause_analysis'] = cause_analysis
        
        return result

    # 나머지 메서드들은 기존과 동일...
    def _get_metadata_sample(self):
        """메타데이터 구조 샘플 가져오기 (한 번만 실행)"""
        if self.metadata_sample is None:
            try:
                collection = self.vectorstore._collection
                sample_data = collection.get(limit=1, include=["metadatas"])
                if sample_data and sample_data.get('metadatas'):
                    self.metadata_sample = sample_data['metadatas'][0]
                else:
                    self.metadata_sample = {}
            except Exception as e:
                print(f"⚠️ 메타데이터 샘플 로드 실패: {e}")
                self.metadata_sample = {}
        return self.metadata_sample
    
    def _get_actual_metadata_values(self, metadatas_sample: List[Dict]) -> Dict[str, List[str]]:
        """실제 메타데이터에서 값들 추출"""
        values = {}
        key_fields = ['ont_food_type', 'ont_food', 'ont_allergen', 'ont_contaminant', 'ont_recall_reason']
        
        for field in key_fields:
            unique_values = set()
            for metadata in metadatas_sample:
                value = metadata.get(field)
                if value and str(value).lower() not in ['null', 'none', '']:
                    unique_values.add(str(value))
            values[field] = list(unique_values)[:10]
        
        return values
    
    def _apply_llm_filter_logic(self, metadatas: List[Dict], documents: List[str], filter_logic: Dict) -> List[Tuple[Dict, str]]:
        """LLM이 생성한 필터링 로직 적용"""
        filtered = []
        combination = filter_logic.get('combination', 'AND')
        
        for metadata, doc in zip(metadatas, documents):
            conditions_met = []
            
            for condition in filter_logic.get('filter_logic', []):
                field = condition['field']
                condition_type = condition['condition']
                value = condition.get('value')
                
                field_value = str(metadata.get(field, '')).lower()
                
                if condition_type == 'contains':
                    met = value.lower() in field_value
                elif condition_type == 'equals':
                    met = field_value == value.lower()
                elif condition_type == 'exists':
                    met = field_value and field_value not in ['null', 'none', '']
                elif condition_type == 'not_exists':
                    met = not field_value or field_value in ['null', 'none', '']
                elif condition_type == 'starts_with':
                    met = field_value.startswith(value.lower())
                else:
                    met = False
                
                conditions_met.append(met)
            
            if combination == 'AND':
                matches = all(conditions_met) if conditions_met else False
            else:
                matches = any(conditions_met) if conditions_met else False
            
            if matches:
                filtered.append((metadata, doc))
        
        return filtered
    
    def is_logical_question(self, question: str) -> bool:
        """질문이 논리형인지 LLM으로 판단"""
        intent = self.analyze_logical_intent(question)
        return intent.get('query_type') == 'logical' and intent.get('confidence', 0) > 0.6
    
    def process_logical_query(self, question: str) -> Dict[str, Any]:
        """논리형 질문 처리 메인 함수"""
        try:
            intent = self.analyze_logical_intent(question)
            
            if intent.get('query_type') != 'logical':
                return {'error': '논리형 질문이 아닙니다.'}
            
            # ChromaDB에서 전체 데이터 가져오기
            print("📊 ChromaDB에서 전체 메타데이터 로딩 중...")
            
            try:
                collection = self.vectorstore._collection
                all_data = collection.get(include=["metadatas", "documents"])
                metadatas = all_data.get('metadatas', [])
                documents = all_data.get('documents', [])
                
                print(f"✅ 총 {len(metadatas)}개 문서의 메타데이터 로드됨")
                
                if not metadatas:
                    return {'error': 'ChromaDB에 데이터가 없습니다.'}
                
            except Exception as db_error:
                print(f"❌ ChromaDB 데이터 로드 실패: {db_error}")
                return {'error': f'ChromaDB 접근 오류: {str(db_error)}'}
            
            # 연산 유형별 처리
            operation = intent.get('operation', 'exclude')
            
            if operation == 'exclude':
                return self._handle_exclude_query_llm(intent, metadatas, documents)
            elif operation == 'compare' or operation == 'temporal':
                return self._handle_compare_query_llm(intent, metadatas, documents)
            elif operation == 'conditional':
                return self._handle_conditional_query_llm(intent, metadatas, documents)
            else:
                return {'error': f'지원하지 않는 논리 연산: {operation}'}
                
        except Exception as e:
            print(f"❌ 논리형 쿼리 처리 오류: {e}")
            return {'error': f'논리형 쿼리 처리 오류: {str(e)}'}