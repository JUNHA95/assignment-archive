"""08 · FastAPI → Agent → Tool → 응답 · 기본필수 · 강사용 정답.

DB: 기존 report_agent_lab / 반드시 SELECT. 실제 Tool은 --live에서 실행됩니다.
"""


# --------------------------------------------------
# [1단계] 기본 질문 준비
# --------------------------------------------------

def default_question() -> str:
    """실제 보고서 검색 + 63번 인덱싱 상태를 묻는 요청 문자열."""

    # demo_question()에 미리 작성된 질문을 가져와 반환함.
    #
    # 예: RAG 근거 인용 정확도 향상 관련 보고서의 내용과
    #     63번 보고서의 인덱싱 상태를 알려줘.
    #
    # 반환된 질문은 main.py에서 Agent를 직접 실행할 때 사용함.

    from coursekit.live import demo_question
    return demo_question()


# --------------------------------------------------
# [2단계] Agent 실행 결과 확인 및 API 응답 구성
# --------------------------------------------------

def api_response(run: dict) -> dict:
    """성공/실패와 최종 답변·출처 검증을 API 계약으로 변환."""

    # ① Agent 실행 완료 여부 확인
    #
    # 경우 A: status가 completed인 경우
    #         → 다음 검증 단계로 이동
    #
    # 경우 B: status가 completed가 아닌 경우
    #         → 오류 발생
    #
    # Agent가 정상적으로 작업을 끝내지 못했는데
    # 사용자에게 성공한 것처럼 응답하는 것을 방지함.

    if run.get('status')!='completed':raise RuntimeError('Agent did not complete')


    # ② 출처 검증 결과 확인
    #
    # 경우 A: citation_ids_valid가 True인 경우
    #         → 최종 응답 구성 단계로 이동
    #
    # 경우 B: citation_ids_valid가 True가 아닌 경우
    #         → 오류 발생
    #
    # 예: 검색을 통해 전달받은 근거 자료의 ID가 14, 46인데
    #     Agent가 전달받지 않은 ID 100을 출처로 인용했다면
    #     출처 검증에 실패하게 됨.
    #
    # 이 함수에서 출처를 직접 검증하는 것은 아니며,
    # 앞선 Agent 실행 과정에서 수행한 검증 결과를 확인함.

    if run.get('citation_ids_valid') is not True:
        raise RuntimeError('UNVERIFIED_CITATION: Agent가 제공되지 않은 출처 ID를 인용했습니다.')


    # ③ 최종 API 응답 구성 및 반환
    #
    # 실행 완료 여부와 출처 검증 결과가 모두 정상인 경우에만
    # 아래 항목들을 딕셔너리로 묶어 반환함.

    return {
        'status':'completed',                               # 실행 완료 상태
        'answer':run['answer'],                             # 최종 답변
        'tool_trace':run['tool_trace'],                     # Tool 사용 기록
        'citation_ids_valid':run['citation_ids_valid'],     # 출처 검증 통과 여부
        'verified_evidence_ids':run['verified_evidence_ids'] # 검증된 근거 ID 목록
    }