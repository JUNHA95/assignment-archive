"""Day21 실습 07: 실패 분류·최대 2회. 실행: python main.py"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import DATA, read_json, write_json, sha_text, run

# 기준: Day21 PPT 56~63장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
from coursekit.retry import run_fixture_scenarios

# ============================================================
# [7교시 전체 흐름] AI 답변의 형식 오류를 수정하고 재시도하기
#
# 1. AI에게 프롬프트를 전달하고 답변 생성
#    └─ generate()
#
# 2. 생성된 답변이 정해진 규칙을 지켰는지 검사
#    └─ validate()
#
# 3. 검사 결과에 따라 처리
#    ├─ 오류 없음 → 정상 종료
#    ├─ JSON·Schema 오류 → repair_prompt()로 수정 요청 생성
#    └─ 그 외 오류 → 자동 수정하지 않고 종료
#
# 4. 수정된 프롬프트로 한 번 더 답변 생성 및 검사
#    ├─ 통과 → 검증된 답변 반환
#    └─ 실패 → None 반환
#
# ※ 최초 1회 + 재시도 1회 = 최대 2번 생성
# ※ 자동 검증 통과는 답변의 사실적 정확성을 보장하지 않음
# ============================================================


def repair_prompt(original, error):
    # [함수 역할]
    # AI 답변에서 JSON 또는 Schema 형식 오류가 발생했을 때,
    # 기존 프롬프트에 오류 수정 지시문을 추가하는 함수
    #
    # 입력: 기존 프롬프트(original), 발생한 오류(error)
    # 출력: 수정 요청이 추가된 프롬프트(repaired)

    # [1단계] 자동 수정이 허용된 오류와 수정 지시문 정의
    # json_error: JSON 문법이 잘못된 경우
    # schema_error: 필수 항목·자료형·길이 등이 규칙에 맞지 않는 경우
    safe = {
        "json_error": "JSON 객체 하나를 코드 블록 없이 출력하세요.",
        "schema_error": "필수 필드·자료형·길이를 지키세요. source_ids는 양의 정수 목록입니다."
    }

    # [2단계] 자동 수정이 허용된 오류인지 확인
    # 예: unknown_source(존재하지 않는 출처)는 자동 수정하지 않음
    # 허용되지 않은 오류라면 ValueError 발생
    if error not in safe:
        raise ValueError("형식 오류만 수정 요청을 만듭니다")

    # [3단계] 기존 프롬프트에 오류 수정 지시문 추가
    # 질문·지식·출력 규칙은 유지하고, 잘못된 형식만 고치도록 요청
    # 부족한 정보를 AI가 임의로 만들어내지 않도록 제한
    repaired = original + "\n[수정 요청] 같은 질문·지식·계약을 유지하세요. " + safe[error] + " 부족한 사실은 만들지 마세요."

    # [4단계] 수정 요청이 추가된 프롬프트 반환
    return repaired


def run_with_retry(generate, prompt, validate):
    # [함수 역할]
    # AI에게 답변을 생성하도록 요청하고 결과를 검사한 뒤,
    # JSON·Schema 오류가 있으면 수정 요청을 추가하여 한 번 더 시도하는 함수
    #
    # 입력:
    # generate: 프롬프트를 전달하면 AI 답변을 생성하는 함수
    # prompt: AI에게 전달할 프롬프트
    # validate: 생성된 답변이 규칙에 맞는지 검사하는 함수
    #
    # 출력:
    # 검증된 답변(obj) 또는 None, 시도별 실행 기록(history)

    # [1단계] 각 시도의 실행 결과를 저장할 빈 리스트 준비
    history = []

    # [2단계] 답변 생성과 검사를 최대 2번 반복
    # attempt=1: 최초 실행
    # attempt=2: 형식 오류 수정 후 재시도
    for attempt in (1, 2):

        # [3단계] 현재 프롬프트를 이용해 AI 답변 생성
        raw = generate(prompt)

        # [4단계] 생성된 답변 검증
        # obj: 검증에 통과한 답변 객체(실패 시 None)
        # error: 발생한 오류 이름(통과 시 None)
        obj, error = validate(raw)

        # [5단계] 현재 시도의 결과 기록
        # attempt: 몇 번째 시도인지
        # error: 검증 오류
        # raw_text: AI가 생성한 원본 답변
        # prompt_sha256: 사용한 프롬프트의 고유 해시값
        # 해시값은 프롬프트 내용을 구분하는 데 사용
        history.append({
            "attempt": attempt,
            "error": error,
            "raw_text": raw,
            "prompt_sha256": sha_text(prompt)
        })

        # [6단계] 검증에 통과하면 즉시 종료
        # 정보 부족 답변도 규칙에 맞으면 정상 결과로 인정
        if error is None:
            return obj, history

        # [7단계] 재시도 가능 여부 확인
        # JSON·Schema 이외의 오류이거나 이미 2번째 시도라면 종료
        if error not in {"json_error", "schema_error"} or attempt == 2:
            break

        # [8단계] 형식 오류 수정 요청을 기존 프롬프트에 추가
        # 다음 반복에서 수정된 프롬프트로 다시 답변 생성
        prompt = repair_prompt(prompt, error)

    # [9단계] 모든 시도가 실패하면 None과 실행 기록 반환
    return None, history

def main():
    fixtures=read_json(DATA/"validation_fixtures.json")
    # 해설 D21-07-B03: 네 fixture 순서의 호출 수 2·2·1·1을 확인하고 기록하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 복구 성공·연속 실패·규칙 중단·부족 정상 종료 네 경우.
    # BEGIN D21-07-B03
    results=run_fixture_scenarios(fixtures,runner=run_with_retry)
    assert [x["calls"] for x in results]==[2,2,1,1]
    write_json(HERE/"outputs"/"retry_results.json",{"source_mode":"fixture","results":results})
    # END D21-07-B03
    print("[07] 제어 흐름만 검증했습니다. 실제 모델 복구율은 측정하지 않았습니다.")

if __name__ == "__main__":
    run(main)
