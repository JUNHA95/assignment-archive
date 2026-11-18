"""Day23 실습 08 · AI Agent용 Retriever (기본필수, mentor).

실행: python main.py --mode fixture
실제 E5/DB: python main.py --mode live
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))

from time import perf_counter
from day23kit.fixtures import fixture_search
from day23kit.embedding import get_encoder, encode_question
from day23kit.db import connect_db, exact_topk, check_data_ready
from day23kit.io import save_result


# BEGIN D23-08-B01
def validate_question(question):
    """
    [기본 1] 검색할 질문이 올바른지 검사

    도서관 직원이 책을 찾아주기 전에
    이용자가 무엇을 찾는지 제대로 말했는지 확인하는 과정
    """

    # [1단계] 질문이 문자열인지, 실제 내용이 있는지 확인
    # 경우 ① "서버는 어디에서 실행하나요?" → 정상
    # 경우 ② "   " 또는 "" → 질문 내용이 없으므로 오류
    # 경우 ③ None 또는 123 → 문자열이 아니므로 오류
    if not isinstance(question, str) or not question.strip():
        raise ValueError('empty_question')

    # [2단계] 질문 앞뒤의 불필요한 공백을 제거하고 반환
    # 예: "  서버는 어디에서 실행하나요?  " → "서버는 어디에서 실행하나요?"
    return question.strip()
# END D23-08-B01


# BEGIN D23-08-B02
def retrieve_candidates(question, mode, k, category, env_file):
    """
    [기본 2] 질문과 관련 있는 문서를 검색

    도서관에서 이용자의 질문에 맞는 책을 찾아주는 과정
    - question: 검색할 질문
    - mode: 모의 검색(fixture) 또는 실제 검색(live)
    - k: 가져올 문서의 최대 개수
    - category: 검색할 문서의 종류 (없으면 전체)
    """

    # [1단계] 검색 방식 선택
    # 경우 ① fixture → 미리 준비된 모의 데이터에서 검색
    # 경우 ② live → 실제 E5 모델과 PostgreSQL에서 검색
    if mode == 'fixture':
        return fixture_search(question, k, category)

    # [2단계] 실제 검색이라면 질문을 벡터(숫자 목록)로 변환
    # E5 모델은 질문의 의미를 숫자로 표현하여 문서와 비교할 수 있게 함
    encoder = get_encoder('live')
    qvec = encode_question(encoder, question)

    # [3단계] PostgreSQL에 연결하여 질문과 가까운 문서 검색
    # k=3이면 가까운 문서를 최대 3개 가져옴
    # category가 지정되면 해당 종류의 문서만 검색
    # DB 연결에 실패하면 빈 목록으로 처리하지 않고 오류를 전달
    with connect_db(env_file) as conn:
        check_data_ready(conn)
        return exact_topk(conn, qvec, k, category)
# END D23-08-B02


# BEGIN D23-08-B03
def format_candidates(hits):
    """
    [기본 3] 검색된 문서를 정해진 형식으로 정리

    도서관에서 찾은 책의 제목, 내용, 분류 등을
    목록으로 정리하여 이용자에게 보여줄 준비를 하는 과정
    """

    # [1단계] 검색된 문서마다 필요한 정보만 추출
    # id: 문서 번호 / title: 제목 / content: 원문 내용
    # category: 문서 종류 / source_version: 자료 버전
    # distance: 질문과 문서 사이의 거리 (작을수록 가까움)
    #
    # 경우 ① 검색된 문서가 있으면 → 문서별 정보 목록 반환
    # 경우 ② 검색된 문서가 없으면 → 빈 목록 [] 반환
    #
    # 원문 내용은 변경하지 않으며, 검색되었다고 정답으로 확정하지 않음
    return [
        {
            'id': h['id'],
            'title': h['title'],
            'content': h['content'],
            'category': h['category'],
            'source_version': h['source_version'],
            'distance': round(float(h['distance']), 6)
        }
        for h in hits
    ]
# END D23-08-B03


def run(mode, level, args):

    # [1단계] 입력된 질문을 검사하고 검색 조건 준비
    # 기본(level=1): 최대 3개 검색, 문서 종류 제한 없음
    # 응용 이상(level>=2): 사용자가 지정한 검색 개수와 종류 적용
    question = validate_question(
        getattr(args, 'question', '서버는 어디에서 실행하나요?')
    )

    k = 3 if level == 1 else args.k
    category = None if level == 1 else args.category

    if level >= 2:
        options = validate_options(k, category)
    else:
        options = {'k': k, 'category': category}

    # [2단계] 검색 실행 및 소요 시간 측정
    # fixture → 모의 검색 / live → 실제 E5·PostgreSQL 검색
    t0 = perf_counter()

    hits = retrieve_candidates(
        question,
        mode,
        options['k'],
        options['category'],
        Path(__file__).with_name('.env')
    )

    elapsed = (perf_counter() - t0) * 1000

    # [3단계] 검색된 문서를 정해진 형식으로 정리
    formatted = format_candidates(hits)

    # [4단계] 검색 결과 구성
    # 검색 문서, 문서 개수, 검색 조건 등을 정리
    # 아직 사람이 근거를 검토하거나 LLM이 답변을 생성한 단계는 아님
    result = dict(
        exercise=8,
        level=level,
        source_mode='simulated' if mode == 'fixture' else 'pgvector_exact',
        question=question,
        options=options,
        retrieved=formatted,
        retrieved_count=len(formatted),
        human_evidence_ok=None,
        next_step='검토 후 Context 후보로 전달; LLM 답변 생성은 Day25 RAG에서 수행'
    )

    # [5단계] 실습 수준에 따른 추가 검사
    # level=1 → 기본 검색 결과만 구성
    # level>=2 → 검색 실행 보고서 추가
    # level>=3 → 모델 및 문서 버전 확인
    if level >= 2:
        result['run_report'] = make_run_report(
            question, hits, mode, elapsed
        )

    if level >= 3:
        result['contract_check'] = check_model_and_versions(hits, mode)

    return result


def main():

    # [1단계] 터미널에서 입력받을 실행 옵션 설정
    # 예: python main.py --mode live
    parser = argparse.ArgumentParser(
        description='Day23 실습 08 · AI Agent용 Retriever'
    )

    parser.add_argument(
        '--mode',
        choices=['fixture', 'live'],
        default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계'
    )

    parser.add_argument(
        '--question',
        default='서버는 어디에서 실행하나요?',
        help='검색 질문'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='반환할 최대 검색 후보 (1~9)'
    )

    parser.add_argument(
        '--category',
        default=None,
        choices=['course', 'development', 'support'],
        help='Metadata 분류 필터'
    )

    args = parser.parse_args()

    # [2단계] 기본 검색 실행
    result = run(args.mode, 1, args)

    # [3단계] 검색 결과를 JSON 파일에 저장하고 터미널에 출력
    path = save_result(
        Path(__file__).resolve().parent,
        'exercise_08.json',
        result
    )

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print('결과 저장:', path)


if __name__ == '__main__':
    main()