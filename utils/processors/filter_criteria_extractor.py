# utils/processors/filter_criteria_extractor.py - 이 코드는 자연어 질문에서 필터링 조건을 지능적으로 추출합니다

import json
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class FilterCriteriaExtractor:
    """사용자 질문에서 ChromaDB 메타데이터 필터링 조건을 지능적으로 추출하는 클래스"""
    
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        # 메타데이터 필드별 매핑 정보
        self.field_mappings = {
            "recall_class": {
                "field": "class",
                "values": ["I", "II", "III"],
                "keywords": ["클래스", "등급", "class", "급", "심각도"]
            },
            "ont_food_type": {
                "field": "ont_food_type", 
                "values": ["Seafood Products", "Livestock Products", "Dairy Products", "Bakery Products", "Beverages"],
                "keywords": ["수산물", "해산물", "육류", "유제품", "빵류", "과자류", "음료", "seafood", "livestock", "dairy", "bakery"]
            },
            "ont_product_state": {
                "field": "ont_product_state",
                "values": ["Refrigerated", "Frozen", "Cooked Food", "Raw", "Dried"],
                "keywords": ["냉장", "냉동", "조리", "익힌", "생", "건조", "refrigerated", "frozen", "cooked", "raw", "dried"]
            },
            "ont_recall_reason": {
                "field": "ont_recall_reason",
                "values": ["Bacterial Contamination", "Allergen Mislabeling", "Foreign Object", "Chemical Contamination"],
                "keywords": ["세균", "박테리아", "알레르겐", "라벨링", "이물질", "화학", "bacterial", "allergen", "labeling", "foreign", "chemical"]
            },
            "ont_contaminant": {
                "field": "ont_contaminant",
                "values": ["Salmonella", "Listeria", "E.coli", "Clostridium botulinum"],
                "keywords": ["살모넬라", "리스테리아", "대장균", "보툴리누스", "salmonella", "listeria", "e.coli", "botulinum"]
            },
            "ont_allergen": {
                "field": "ont_allergen", 
                "values": ["Milk", "Eggs", "Peanuts", "Tree nuts", "Soy", "Wheat", "Fish", "Shellfish"],
                "keywords": ["우유", "계란", "땅콩", "견과류", "콩", "밀", "생선", "갑각류", "milk", "eggs", "peanuts", "nuts", "soy", "wheat", "fish", "shellfish"]
            },
            "effective_date": {
                "field": "effective_date",
                "values": [],
                "keywords": ["날짜", "기간", "최근", "이전", "후", "date", "recent", "before", "after"]
            }
        }
    
    def extract_filter_conditions(self, question: str) -> Dict[str, Any]:
        """사용자 질문에서 필터링 조건을 추출하는 메인 함수 - 논리형 질문 제외"""
        try:
            # 🆕 빠른 논리형 질문 체크
            comparison_keywords = ["비교", "compare", "대비", "vs", "중 어느", "더 많아", "차이", "년과", "작년과 올해"]
            question_lower = question.lower()
            
            if any(keyword in question_lower for keyword in comparison_keywords):
                print(f"🔍 논리형 질문 감지 - 필터링 건너뜀: {question}")
                return {
                    "filters_detected": False,
                    "conditions": {},
                    "confidence": 0.0,
                    "extraction_method": "logical_question_detected"
                }
            
            # 기존 LLM 및 패턴 추출 로직
            llm_result = self._extract_with_llm(question)
            pattern_result = self._extract_with_patterns(question)
            
            final_conditions = self._merge_extraction_results(llm_result, pattern_result)
            
            has_filters = bool(final_conditions.get('conditions'))
            
            print(f"🔍 필터 조건 추출 결과: {has_filters} - {final_conditions.get('conditions', {})}")
            
            return {
                "filters_detected": has_filters,
                "conditions": final_conditions.get('conditions', {}),
                "confidence": final_conditions.get('confidence', 0.5),
                "extraction_method": final_conditions.get('method', 'hybrid')
            }
            
        except Exception as e:
            print(f"⚠️ 필터 조건 추출 오류: {e}")
            return {
                "filters_detected": False,
                "conditions": {},
                "confidence": 0.0,
                "extraction_method": "error"
            }
    
    def _extract_with_llm(self, question: str) -> Dict[str, Any]:
        """LLM을 활용한 스마트 필터 조건 추출 - 복합 조건 지원"""
        try:
            metadata_info = self._generate_metadata_guide()
            
            prompt = f"""
    다음 리콜 관련 질문을 분석하여 ChromaDB 메타데이터 필터링 조건을 JSON 형태로 추출하세요.
    반드시 JSON만 반환하고 다른 설명은 하지 마세요.

    질문: "{question}"

    🚨 질문 분류 규칙:
    1. **필터링 질문 (is_filtering: true)**:
    - "해산물 리콜만", "클래스 I만", "살모넬라 관련만"
    - "A 중에서 B인 사례", "A에서 B 또는 C가 원인"
    - 특정 조건으로 데이터를 걸러내는 질문

    2. **논리형 질문 (is_filtering: false)**:
    - "A와 B 비교", "A 대비 B", "어느 쪽이 더"
    - "A이면서 B가 아닌", "A를 제외한 B"

    3. **수치형 질문 (is_filtering: false)**:
    - "몇 건", "상위 N개", "가장 많은"

    사용 가능한 메타데이터 필드와 값:
    {metadata_info}

    🔍 추출 예시:
    - "해산물 리콜 중 살모넬라 원인" → {{"ont_food_type": "Seafood Products", "ont_contaminant": "Salmonella"}}
    - "클래스 I 유제품 리콜" → {{"class": "I", "ont_food_type": "Dairy Products"}}
    - "살모넬라 또는 리스테리아" → {{"ont_contaminant": ["Salmonella", "Listeria"]}} (OR 조건)

    출력 형식:
    {{
    "is_filtering": true/false,
    "conditions": {{
        "ont_food_type": "Seafood Products",
        "ont_contaminant": ["Salmonella", "Listeria"]
    }},
    "confidence": 0.9,
    "reasoning": "해산물과 특정 세균(살모넬라, 리스테리아) 조건이 명확히 언급됨"
    }}

    JSON 결과:"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            llm_result = json.loads(result_text)
            
            # is_filtering이 false면 조건 비우기
            if not llm_result.get('is_filtering', True):
                llm_result['conditions'] = {}
                llm_result['confidence'] = 0.0
            
            llm_result['method'] = 'llm'
            
            print(f"🧠 LLM 추출 결과: {llm_result.get('conditions', {})} (필터링여부: {llm_result.get('is_filtering', True)})")
            return llm_result
            
        except Exception as e:
            print(f"⚠️ LLM 추출 실패: {e}")
            return {"conditions": {}, "confidence": 0.0, "method": "llm_failed"}
        
    def _extract_with_patterns(self, question: str) -> Dict[str, Any]:
        """패턴 기반 백업 필터 조건 추출 - 복합 조건 지원"""
        conditions = {}
        confidence_score = 0.0
        
        question_lower = question.lower()
        
        # 🆕 복합 조건 처리
        # 해산물 관련
        if any(word in question_lower for word in ["해산물", "수산물", "seafood", "생선", "새우", "게"]):
            conditions["ont_food_type"] = "Seafood Products"
            confidence_score += 0.3
        
        # 육류 관련  
        if any(word in question_lower for word in ["육류", "고기", "beef", "pork", "chicken", "meat"]):
            conditions["ont_food_type"] = "Livestock Products"
            confidence_score += 0.3
        
        # 유제품 관련
        if any(word in question_lower for word in ["유제품", "우유", "치즈", "dairy", "milk", "cheese"]):
            conditions["ont_food_type"] = "Dairy Products"
            confidence_score += 0.3
        
        # 🆕 세균 관련 - OR 조건 처리
        bacteria_found = []
        if any(word in question_lower for word in ["살모넬라", "salmonella"]):
            bacteria_found.append("Salmonella")
        if any(word in question_lower for word in ["리스테리아", "listeria"]):
            bacteria_found.append("Listeria")
        if any(word in question_lower for word in ["대장균", "e.coli", "ecoli"]):
            bacteria_found.append("E.coli")
        
        if bacteria_found:
            if len(bacteria_found) == 1:
                conditions["ont_contaminant"] = bacteria_found[0]
            else:
                conditions["ont_contaminant"] = bacteria_found  # 복수 조건
            confidence_score += 0.4
        
        # 클래스 관련
        class_match = re.search(r'(?:class|클래스|등급)\s*([I1-3])', question_lower)
        if class_match:
            class_num = class_match.group(1)
            if class_num in ['1', 'I']:
                conditions["class"] = "I"
            elif class_num in ['2', 'II']:
                conditions["class"] = "II"
            elif class_num in ['3', 'III']:
                conditions["class"] = "III"
            confidence_score += 0.3
        
        print(f"🔧 패턴 추출 결과: {conditions}")
        
        return {
            "conditions": conditions,
            "confidence": min(confidence_score, 1.0),
            "method": "pattern"
        }
    
    def _merge_extraction_results(self, llm_result: Dict, pattern_result: Dict) -> Dict[str, Any]:
        """LLM 결과와 패턴 결과를 지능적으로 통합"""
        llm_conditions = llm_result.get('conditions', {})
        pattern_conditions = pattern_result.get('conditions', {})
        
        llm_confidence = llm_result.get('confidence', 0.0)
        pattern_confidence = pattern_result.get('confidence', 0.0)
        
        # LLM 결과를 우선시하되, 패턴 결과로 보완
        merged_conditions = llm_conditions.copy()
        
        # 패턴에서만 발견된 조건 추가 (LLM에서 누락된 경우)
        for field, value in pattern_conditions.items():
            if field not in merged_conditions:
                merged_conditions[field] = value
        
        # 신뢰도는 더 높은 쪽 + 보너스
        final_confidence = max(llm_confidence, pattern_confidence)
        if llm_conditions and pattern_conditions:
            final_confidence = min(final_confidence + 0.1, 1.0)  # 두 방법 모두 결과가 있으면 보너스
        
        return {
            "conditions": merged_conditions,
            "confidence": final_confidence,
            "method": "hybrid"
        }
    
    def _generate_metadata_guide(self) -> str:
        """LLM에게 제공할 메타데이터 필드 가이드 생성"""
        guide_lines = []
        
        for field_key, field_info in self.field_mappings.items():
            field_name = field_info["field"]
            values = field_info["values"]
            keywords = field_info["keywords"]
            
            if values:
                value_list = ", ".join(values[:5])  # 상위 5개만 표시
                guide_lines.append(f"- {field_name}: {value_list} (키워드: {', '.join(keywords[:3])})")
            else:
                guide_lines.append(f"- {field_name}: 날짜 형식 (키워드: {', '.join(keywords[:3])})")
        
        return "\n".join(guide_lines)
    
    def is_filtering_question(self, question: str) -> bool:
        """질문이 필터링 질문인지 빠르게 판단"""
        result = self.extract_filter_conditions(question)
        return result.get('filters_detected', False) and result.get('confidence', 0) > 0.6
    
    def convert_to_chroma_filter(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """추출된 조건을 ChromaDB 필터 형식으로 변환 - OR 조건 지원"""
        if not conditions:
            return {}
        
        chroma_filter = {}
        
        for field, value in conditions.items():
            if value and str(value).strip():
                if isinstance(value, list):
                    # OR 조건 - ChromaDB의 $in 연산자 사용
                    chroma_filter[field] = {"$in": value}
                else:
                    chroma_filter[field] = value
        
        print(f"🔧 ChromaDB 필터 변환: {chroma_filter}")
        return chroma_filter