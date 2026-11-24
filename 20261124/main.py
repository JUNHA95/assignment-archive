"""
[6교시] 기존 LangChain RAG에 Rewrite와 Reranker 연결

[실습 목적]
기존 RAG 파이프라인에 Rewrite, Multi Query, Reranking 기능을 연결하고,
필요에 따라 각 기능의 사용 여부를 설정하는 실습이다.

[전체 실행 구조]
1. 파이프라인 입력 및 실행 조건 설정
2. 각 단계에서 사용할 모델과 함수 준비
3. 각 기능을 하나의 파이프라인으로 연결
4. 파이프라인 실행
5. 실행 결과 저장

[친숙한 사례: 쇼핑몰 고객센터 AI]
사용자가 이전 대화에서 상품을 반품하고 싶다고 말한 후
"그럼 배송비는 얼마야?"라고 질문했다고 가정한다.

Rewrite   : "상품 반품 시 배송비는 얼마인가요?"로 질문 재작성
Multi Query: 필요하다면 다양한 표현의 검색 질문 추가 생성
Retrieval : 반품 및 배송비 관련 후보 문서 검색
Reranking : 질문과 후보 문서의 관련성을 재평가하여 순위 조정
Context   : 답변에 적합한 문서를 선택하고 정리
Prompt    : 답변 규칙 + 질문 + Context 결합
Generation: LLM을 이용하여 답변 생성
Validation: 답변 형식 및 출처 검증
Response  : 답변, 출처, 처리 이력 반환
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROLE_ROOT = next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0, str(ROLE_ROOT))

import argparse

from coursekit.workshops import STAGE_RUNNERS
from coursekit.io import save_report


# ============================================================
# [1단계] 파이프라인 입력 및 실행 조건 설정
# ============================================================

def prepare_pipeline_request(case):

    # 파이프라인에 전달할 질문, 검색 조건, 기능 사용 여부를 설정한다.
    # 원본 질문은 그대로 보존하고, 검색용 질문은 Rewrite 단계에서 생성한다.

    # BEGIN D27-06-B01
    return {
        'query': case['question'],       # 사용자의 원본 질문
        'history': case['history'],      # 이전 대화 내용
        'category': case['category'],    # 검색할 문서의 카테고리

        'candidate_k': 5,                # 검색 질문당 최대 후보 문서 수
        'context_k': 2,                  # Context에 포함할 최대 문서 수

        'use_rewrite': True,             # 검색용 질문 재작성 사용
        'use_multiquery': False,         # 추가 검색 질문 생성 미사용
        'use_rerank': True               # 후보 문서 순위 재조정 사용
    }
    # END D27-06-B01


# ============================================================
# [2단계] 각 기능을 하나의 RAG 파이프라인으로 연결
# ============================================================

def wire_langchain_pipeline(dependencies, offline):

    # dependencies:
    # 각 단계에서 사용할 검색, 질문 재작성, 순위 조정, 답변 생성 함수.
    #
    # runtime.py의 create_dependencies()에서 준비한다.
    #
    # [경우 1] offline=True
    # - 미리 준비된 검색 결과, 재작성 질문, 관련성 점수 사용
    # - 실제 LLM을 호출하지 않고 전체 코드의 실행 흐름 확인
    #
    # [경우 2] offline=False
    # - Rewrite: 실제 LLM 사용
    # - Retrieval: E5 Embedding 모델 및 PostgreSQL 사용
    # - Reranking: Cross Encoder 사용
    # - Generation: 실제 LLM 사용

    # BEGIN D27-06-B02
    from coursekit.pipeline import create_rag_pipeline

    # create_rag_pipeline()은 준비된 기능을 순서대로 연결한다.
    # 이 시점에는 파이프라인을 구성할 뿐, 실제 실행하지 않는다.

    # --------------------------------------------------------
    # [파이프라인 내부의 실행 순서]
    # 파일 위치: coursekit/pipeline.py
    # --------------------------------------------------------
    #
    # ① start(): 입력 검증 및 초기 상태 설정
    #
    #    - 질문, History, 검색 조건 등이 올바른 형식인지 확인한다.
    #    - 각 단계의 처리 결과를 저장할 딕셔너리(s)를 준비한다.
    #
    #
    # ② rewrite(): 검색용 질문 생성
    #
    #    주요 함수: rewrite_query()
    #
    #    [사용하는 경우]
    #    원본 질문과 History를 이용하여 검색용 질문을 생성한다.
    #
    #    예: "그럼 배송비는 얼마야?"
    #        → "상품 반품 시 배송비는 얼마인가요?"
    #
    #    [사용하지 않는 경우]
    #    원본 질문을 그대로 검색용 질문으로 사용한다.
    #
    #    [질문의 의미를 파악할 수 없는 경우]
    #    clarification_required 상태로 변경한다.
    #    이후 검색, Reranking, Context 구성은 건너뛰고
    #    추가 정보가 필요하다는 응답을 반환한다.
    #
    #
    # ③ queries(): 검색 질문 목록 구성
    #
    #    주요 함수: unique_queries()
    #
    #    [Multi Query를 사용하는 경우]
    #    검색용 질문을 바탕으로 추가 질문을 생성하고 중복을 제거한다.
    #
    #    예: "상품 반품 시 배송비는 얼마인가요?"
    #        "반품 배송비 부담 기준은?"
    #        "상품 반환 시 발생하는 비용은?"
    #
    #    [Multi Query를 사용하지 않는 경우]
    #    검색용 질문 하나만 사용한다.
    #
    #    ※ 현재 실습은 use_multiquery=False이며,
    #      추가 질문 생성 함수도 별도로 연결하지 않는다.
    #
    #
    # ④ retrieve(): 후보 문서 검색
    #
    #    주요 함수:
    #    - search_multiple(): 검색 질문별 검색 실행 및 결과 통합
    #    - make_live_search(): 실제 검색 함수(search) 준비
    #    - encode(): 검색 질문을 벡터로 변환
    #    - search_db(): PostgreSQL에서 관련 문서 검색
    #    - hit_to_document(): 검색 결과를 Document 형식으로 변환
    #    - merge_found(): 검색 결과 통합 및 중복 문서 제거
    #
    #    검색 질문을 벡터로 변환하고 DB에서 관련 문서를 검색한다.
    #    이때 카테고리, 사용자 권한 등의 검색 조건을 적용한다.
    #
    #    예: 반품 정책, 배송비 규정, 교환 정책 등 후보 문서 확보
    #
    #    candidate_k=5이므로 검색 질문당 최대 5개를 검색한다.
    #
    #
    # ⑤ rerank(): 후보 문서 순위 재조정
    #
    #    주요 함수:
    #    - live_scorer(): Cross Encoder로 관련성 점수 계산
    #    - rerank_documents(): 점수를 기준으로 문서 순위 변경
    #
    #    [사용하는 경우]
    #    질문과 후보 문서 본문을 함께 Cross Encoder에 입력하여
    #    관련성을 정밀하게 평가하고 점수가 높은 순서대로 정렬한다.
    #
    #    예: 일반 반품 절차보다 반품 배송비 규정의 순위를 높인다.
    #
    #    [사용하지 않는 경우]
    #    Retrieval에서 확보한 후보 문서의 기존 순서를 유지한다.
    #
    #    [후보 문서가 없는 경우]
    #    Reranking을 수행하지 않는다.
    #
    #
    # ⑥ context(): 최종 문서 선택 및 Context 구성
    #
    #    주요 함수: context_from_documents()
    #
    #    - 순위가 높은 문서부터 확인한다.
    #    - 중복 문서를 제외한다.
    #    - 출처 ID, 출처 위치, 문서 본문을 하나로 묶는다.
    #    - Context 전체 길이가 650자를 넘지 않도록 제한한다.
    #    - context_k=2이므로 최대 2개의 문서를 선택한다.
    #
    #    예: 반품 배송비 규정과 상품 불량 시 배송비 규정을 선택한다.
    #
    #
    # ⑦ generate(): Prompt 구성, 답변 생성, 검증 및 출처 정리
    #
    #    주요 함수:
    #    - create_generation(): Prompt와 LLM을 연결하는 Chain 준비
    #    - validate_generation(): 생성된 답변 검증
    #    - source_index(): 출처 ID와 상세 출처 정보 연결
    #    - make_trace(): 각 단계의 처리 이력 정리
    #
    #    [Context가 충분한 경우]
    #    답변 규칙 + 원본 질문 + 검색용 질문 + Context를
    #    Prompt로 구성하여 LLM에 전달한다.
    #
    #    예: "단순 변심으로 반품하는 경우 배송비는 3,000원이며,
    #         상품 불량으로 반품하는 경우에는 무료입니다."
    #
    #    이후 답변 형식과 출처 ID 등을 검증한다.
    #    단, 답변 내용이 근거와 의미상 일치하는지까지 검증하지는 않는다.
    #
    #    [Context가 없는 경우]
    #    LLM을 호출하지 않고 근거가 부족하다는 응답을 반환한다.
    #
    #    [Offline인 경우]
    #    실제 LLM 대신 교육용 답변 생성 함수(offline_answer)를 사용한다.
    #
    #    최종적으로 답변, 출처 상세 정보, 처리 이력을 반환한다.
    #
    # --------------------------------------------------------
    # [각 단계의 연결 방식]
    #
    # LangChain의 LCEL 문법(|)을 사용하여 함수를 연결한다.
    #
    # start → rewrite → queries → retrieve → rerank
    #       → context → generate
    #
    # 각 단계는 처리 결과를 딕셔너리(s)에 추가하고,
    # 해당 딕셔너리를 다음 단계로 전달한다.
    #
    # 따라서 원본 질문, 검색용 질문, 후보 문서, 재정렬된 문서,
    # Context 등의 정보가 단계별로 누적된다.
    #
    # run(): 각 단계의 실행 시간 측정 및 오류 처리
    # make_trace(): 검색 질문, 문서 ID, 실행 시간 등 처리 이력 기록
    # --------------------------------------------------------

    return create_rag_pipeline(*dependencies, offline=offline)
    # END D27-06-B02


# ============================================================
# [3단계] 실행 방식 설정 및 전체 파이프라인 실행
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description='Day27 실습 06 · 06_langchain_pipeline'
    )

    # python main.py --offline : 교육용 고정 데이터 사용
    # python main.py           : 실제 모델, DB, LLM 사용
    parser.add_argument(
        '--offline',
        action='store_true',
        help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.'
    )

    args = parser.parse_args()

    # STAGE_RUNNERS[6]은 workshops.py의 run_06()을 호출한다.
    #
    # [run_06() 내부의 실행 순서]
    #
    # ① get_question('Q02')
    #    실습에 사용할 질문과 History 등을 불러온다.
    #
    # ② prepare_pipeline_request(case)
    #    파이프라인 입력값과 각 기능의 사용 여부를 설정한다.
    #
    # ③ create_dependencies(offline)
    #    실행 방식에 맞는 모델과 함수를 준비한다.
    #
    # ④ wire_langchain_pipeline(dependencies, offline)
    #    준비된 기능을 하나의 파이프라인으로 연결한다.
    #
    # ⑤ chain.invoke(req)
    #    입력값을 전달하여 연결된 파이프라인을 실제로 실행한다.
    #
    # ⑥ 실행 결과 반환
    #    답변 상태, 답변 내용, 출처, 처리 이력을 반환한다.

    report = STAGE_RUNNERS[6](
        args.offline,
        [prepare_pipeline_request, wire_langchain_pipeline]
    )


    # ========================================================
    # [4단계] 실행 결과 저장
    # ========================================================

    # 실행 방식에 따라 결과를 서로 다른 JSON 파일로 저장한다.
    #
    # Offline: day27_06_fixture.json
    # Live   : day27_06_live.json

    path = save_report(
        HERE,
        'day27_06_' + ('fixture' if args.offline else 'live') + '.json',
        report
    )

    print('실습 06 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')


if __name__ == '__main__':
    main()