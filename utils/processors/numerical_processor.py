# utils/processors/numerical_processor.py - 이 코드는 수치형 질문 처리만 담당합니다

import json
import re
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class NumericalQueryProcessor:
    """LLM과 ChromaDB 메타데이터를 활용한 스마트한 수치형 질문 처리"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """LLM을 활용한 스마트한 질문 의도 분석"""
        
        prompt = f"""
다음 질문을 분석하여 JSON 형태로 정확히 응답하세요. 반드시 JSON만 반환하고 다른 설명은 하지 마세요.

질문: "{question}"

분석 기준:
1. query_type: 
   - "numerical": 개수, 순위, 비율, 통계가 필요한 질문
   - "semantic": 구체적인 사례나 내용을 찾는 질문
   - "general": 리콜과 무관한 일반 질문

2. operation (numerical인 경우만):
   - "count": 개수 세기 (몇 건, 몇 개, 총 몇)
   - "ranking": 순위 매기기 (상위, 최고, 가장 많은, 빈번)
   - "percentage": 비율 계산 (비율, 퍼센트, 비중)
   - "comparison": 비교 (증가, 감소, 차이)

3. target_field (분석 대상):
   - "company": 회사/업체 관련
   - "food": 제품/식품 관련  
   - "food_type": 식품 유형/카테고리
   - "contaminant": 오염물질
   - "allergen": 알레르기 유발요소
   - "reason": 리콜 이유/원인

4. filter_keywords: 필터링에 사용할 키워드들 (배열)

5. number: 질문에 포함된 숫자 (상위 N개 등)

6. confidence: 분석 확신도 (0.0-1.0)

예시:
- "살모넬라균으로 인한 리콜이 총 몇 건이었어?" 
  → {{"query_type": "numerical", "operation": "count", "filter_keywords": ["살모넬라", "Salmonella"], "confidence": 0.95}}

- "리콜이 가장 빈번한 상위 3개 회사는?"
  → {{"query_type": "numerical", "operation": "ranking", "target_field": "company", "number": 3, "confidence": 0.9}}

- "불닭볶음면 리콜 사례 알려줘"
  → {{"query_type": "semantic", "filter_keywords": ["불닭볶음면"], "confidence": 0.85}}

JSON 결과:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # JSON 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            # JSON 파싱
            intent = json.loads(result_text)
            
            # 기본값 설정
            intent.setdefault('query_type', 'semantic')
            intent.setdefault('confidence', 0.5)
            intent.setdefault('filter_keywords', [])
            
            print(f"🧠 LLM 의도 분석 결과: {intent}")
            return intent
            
        except Exception as e:
            print(f"⚠️ LLM 의도 분석 실패: {e}")
            return self._fallback_pattern_analysis(question)
    
    def _fallback_pattern_analysis(self, question: str) -> Dict[str, Any]:
        """LLM 실패 시 폴백용 패턴 분석"""
        question_lower = question.lower()
        
        # 수치형 패턴 체크
        numerical_patterns = ['몇 건', '몇개', '총 몇', '상위', '최고', '가장 많은', '비율', '퍼센트']
        is_numerical = any(pattern in question_lower for pattern in numerical_patterns)
        
        if is_numerical:
            # 기본 수치형 분류
            if any(p in question_lower for p in ['몇 건', '몇개', '총 몇']):
                operation = 'count'
            elif any(p in question_lower for p in ['상위', '최고', '가장 많은']):
                operation = 'ranking'
            else:
                operation = 'count'
            
            return {
                'query_type': 'numerical',
                'operation': operation,
                'confidence': 0.6,
                'filter_keywords': []
            }
        else:
            return {
                'query_type': 'semantic',
                'confidence': 0.7,
                'filter_keywords': []
            }
    
    def is_numerical_question(self, question: str) -> bool:
        """질문이 수치형인지 LLM으로 판단"""
        intent = self.analyze_question_intent(question)
        return intent.get('query_type') == 'numerical' and intent.get('confidence', 0) > 0.6
    
    def process_numerical_query(self, question: str) -> Dict[str, Any]:
        """LLM 분석 결과를 바탕으로 수치형 질문 처리"""
        try:
            # 의도 분석
            intent = self.analyze_question_intent(question)
            
            if intent.get('query_type') != 'numerical':
                return {'error': '수치형 질문이 아닙니다.'}
            
            # ChromaDB에서 전체 데이터 가져오기
            print("📊 ChromaDB에서 전체 메타데이터 로딩 중...")
            
            try:
                collection = self.vectorstore._collection
                all_data = collection.get(include=["metadatas", "documents"])
                
                metadatas = all_data.get('metadatas', [])
                print(f"✅ 총 {len(metadatas)}개 문서의 메타데이터 로드됨")
                
                if not metadatas:
                    return {'error': 'ChromaDB에 데이터가 없습니다.'}
                
            except Exception as db_error:
                print(f"❌ ChromaDB 데이터 로드 실패: {db_error}")
                return {'error': f'ChromaDB 접근 오류: {str(db_error)}'}
            
            # 연산 유형별 처리
            operation = intent.get('operation', 'count')
            
            if operation == 'count':
                return self._handle_count_query(intent, metadatas)
            elif operation == 'ranking':
                return self._handle_ranking_query(intent, metadatas)
            elif operation == 'percentage':
                return self._handle_percentage_query(intent, metadatas)
            elif operation == 'comparison':
                return self._handle_comparison_query(intent, metadatas)
            else:
                return {'error': f'지원하지 않는 연산: {operation}'}
                
        except Exception as e:
            print(f"❌ 수치형 쿼리 처리 오류: {e}")
            return {'error': f'수치형 쿼리 처리 오류: {str(e)}'}
    
    def _handle_count_query(self, intent: Dict, metadatas: List[Dict]) -> Dict[str, Any]:
        """개수 세기 처리"""
        filtered_data = self._apply_smart_filter(metadatas, intent)
        count = len(filtered_data)
        
        filter_info = self._generate_filter_description(intent)
        
        return {
            'type': 'count',
            'result': count,
            'description': f"{filter_info}조건에 맞는 리콜 건수: {count}건",
            'intent': intent
        }
    
    def _handle_ranking_query(self, intent: Dict, metadatas: List[Dict]) -> Dict[str, Any]:
        """순위 처리"""
        target_field = self._determine_target_field(intent)
        number = intent.get('number', 5)
        
        filtered_data = self._apply_smart_filter(metadatas, intent)
        values = self._extract_field_values(filtered_data, target_field)
        
        counter = Counter(values)
        top_items = counter.most_common(number)
        
        result_list = []
        for item, count in top_items:
            result_list.append({'name': item, 'count': count})
        
        filter_info = self._generate_filter_description(intent)
        
        return {
            'type': 'ranking',
            'result': result_list,
            'description': f"{filter_info}상위 {number}개 {target_field} 순위",
            'total_items': len(result_list),
            'target_field': target_field,
            'intent': intent
        }
    
    def _handle_percentage_query(self, intent: Dict, metadatas: List[Dict]) -> Dict[str, Any]:
        """비율 계산 처리"""
        filtered_data = self._apply_smart_filter(metadatas, intent)
        
        total_count = len(metadatas)
        filtered_count = len(filtered_data)
        
        percentage = (filtered_count / total_count * 100) if total_count > 0 else 0
        
        filter_info = self._generate_filter_description(intent)
        
        return {
            'type': 'percentage',
            'result': round(percentage, 2),
            'description': f"{filter_info}전체 리콜 중 {percentage:.1f}% ({filtered_count}/{total_count}건)",
            'intent': intent
        }
    
    def _handle_comparison_query(self, intent: Dict, metadatas: List[Dict]) -> Dict[str, Any]:
        """비교 분석 처리"""
        filtered_data = self._apply_smart_filter(metadatas, intent)
        
        # 월별 그룹화
        monthly_counts = {}
        for item in filtered_data:
            date_str = item.get('effective_date', '')
            if date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    month_key = date.strftime('%Y-%m')
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                except:
                    continue
        
        if len(monthly_counts) < 2:
            return {'error': '비교할 데이터가 충분하지 않습니다.'}
        
        # 최근 3개월 vs 이전 3개월 비교
        sorted_months = sorted(monthly_counts.keys())
        recent_months = sorted_months[-3:] if len(sorted_months) >= 3 else sorted_months
        earlier_months = sorted_months[-6:-3] if len(sorted_months) >= 6 else sorted_months[:-3]
        
        recent_avg = sum(monthly_counts[m] for m in recent_months) / len(recent_months) if recent_months else 0
        earlier_avg = sum(monthly_counts[m] for m in earlier_months) / len(earlier_months) if earlier_months else 0
        
        if earlier_avg > 0:
            change_rate = ((recent_avg - earlier_avg) / earlier_avg) * 100
            trend = "증가" if change_rate > 5 else "감소" if change_rate < -5 else "유사"
        else:
            change_rate = 0
            trend = "데이터 부족"
        
        filter_info = self._generate_filter_description(intent)
        
        return {
            'type': 'comparison',
            'result': {
                'recent_avg': round(recent_avg, 1),
                'earlier_avg': round(earlier_avg, 1),
                'change_rate': round(change_rate, 1),
                'trend': trend
            },
            'description': f"{filter_info}최근 추세: {trend} (변화율: {change_rate:+.1f}%)",
            'intent': intent
        }
    
    def _apply_smart_filter(self, metadatas: List[Dict], intent: Dict) -> List[Dict]:
        """LLM 분석 결과를 바탕으로 ChromaDB 메타데이터 스마트 필터링"""
        filter_keywords = intent.get('filter_keywords', [])
        
        if not filter_keywords:
            print("🔍 필터 키워드 없음 - 전체 데이터 사용")
            return metadatas
        
        print(f"🔍 필터 키워드: {filter_keywords}")
        
        filtered = []
        for metadata in metadatas:
            if not metadata:
                continue
            
            searchable_fields = [
                'ont_contaminant', 'ont_allergen', 'ont_recall_reason',
                'ont_food', 'ont_food_type', 'title', 'category'
            ]
            
            found_match = False
            for keyword in filter_keywords:
                keyword_lower = keyword.lower()
                
                for field in searchable_fields:
                    field_value = str(metadata.get(field, '')).lower()
                    if field_value and keyword_lower in field_value:
                        found_match = True
                        break
                
                if found_match:
                    break
            
            if found_match:
                filtered.append(metadata)
        
        print(f"📊 필터링 결과: {len(filtered)}/{len(metadatas)}개 문서")
        return filtered
    
    def _determine_target_field(self, intent: Dict) -> str:
        """LLM 분석 결과를 바탕으로 분석 대상 필드 결정"""
        target_field = intent.get('target_field', '')
        
        field_mapping = {
            'company': 'title',
            'food': 'ont_food',
            'food_type': 'ont_food_type',
            'contaminant': 'ont_contaminant',
            'allergen': 'ont_allergen',
            'reason': 'ont_recall_reason'
        }
        
        return field_mapping.get(target_field, 'ont_food_type')
    
    def _extract_field_values(self, metadatas: List[Dict], field: str) -> List[str]:
        """ChromaDB 메타데이터에서 필드 값 추출"""
        values = []
        print(f"🔍 '{field}' 필드에서 값 추출 중...")
        
        for metadata in metadatas:
            value = metadata.get(field, '')
            if value and value != 'null' and str(value).strip():
                if field == 'title':
                    # 회사명 추출
                    company_name = value.split(',')[0].split(' ')[0].strip() if value else ''
                    if company_name and len(company_name) > 1:
                        values.append(company_name)
                else:
                    values.append(str(value))
        
        print(f"✅ 총 {len(values)}개 값 추출됨")
        return values
    
    def _generate_filter_description(self, intent: Dict) -> str:
        """필터 설명 생성"""
        filter_keywords = intent.get('filter_keywords', [])
        if filter_keywords:
            return f"'{', '.join(filter_keywords)}' 관련 "
        return ""
    