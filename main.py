"""
Day21 실습 04: Context 선택·예산
실행: python main.py

[실습 목적]
AI에게 전달할 참고 지식(Context)을 선택하고,
사용 가능한 token 예산을 계산한 뒤 결과를 파일로 저장한다.

[전체 실행 흐름]

main() : 전체 작업을 관리하는 함수
│
├─ 1. 전체 지식 불러오기
│     └─ knowledge_v1.json → rows
│
├─ 2. select_context() : 필요한 지식 선택
│     ├─ 전체 지식을 ID 기준으로 정리
│     ├─ 중복 ID와 존재하지 않는 ID 검사
│     ├─ 지정한 ID(102)의 지식만 선택
│     └─ 선택한 지식을 JSON 문자열로 변환
│
├─ 3. calculate_budget() : token 예산 계산
│     ├─ 전체 token 공간 설정 (8192)
│     ├─ 지시문·질문·예시·답변·여유 공간 제외
│     └─ Context 사용 가능 공간 계산 (4936)
│
├─ 4. Context 구성 기록
│     ├─ 포함한 지식 ID 기록 (102)
│     ├─ 제외한 지식 ID와 이유 기록 (101, 103)
│     └─ token 예산과 함께 정리
│
└─ 5. 결과 파일 저장
      ├─ context.json : 선택한 지식의 원문
      └─ context_manifest.json : 선택·제외 기록 및 예산

[핵심]
필요한 지식 선택 → token 예산 계산 → 선택 내역 저장

※ 실제 AI 모델을 호출하거나 token 수를 측정하는 실습은 아니다.
"""

from pathlib import Path
import sys
import json

# [실행 환경 준비]
# 현재 파일의 위치를 기준으로 경로를 설정하고,
# 수업에서 제공하는 데이터 읽기·쓰기 함수를 불러옴
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import DATA, read_json, write_json, write_text, run


def select_context(rows, selected_ids):
    # [함수 역할]
    # 전체 지식 중 지정한 ID에 해당하는 지식만 선택하고,
    # AI에게 전달할 수 있도록 JSON 문자열로 변환하는 함수
    #
    # 입력: 전체 지식 목록(rows), 선택할 지식 번호(selected_ids)
    # 출력: 선택된 지식 목록(chosen), JSON 문자열(context_json)

    # [1단계] 지식 ID를 기준으로 데이터를 찾기 쉽게 정리
    # 예: [{"id":101,...}, {"id":102,...}]
    #  → {101: {...}, 102: {...}}
    by_id = {row["id"]: row for row in rows}

    # [2단계] 전체 지식에 중복된 ID가 있는지 확인
    # 딕셔너리는 같은 키를 중복 저장할 수 없으므로
    # 원래 데이터 개수와 다르면 중복 ID가 있다는 의미
    if len(by_id) != len(rows):
        raise ValueError("중복 ID를 확인하세요")

    # [3단계] 선택한 ID의 중복 제거
    # 예: [102, 102, 101] → [102, 101]
    # 중복은 제거하지만 선택한 순서는 유지
    ids = list(dict.fromkeys(selected_ids))

    # [4단계] 선택한 ID가 실제 지식 목록에 존재하는지 확인
    # 하나라도 존재하지 않으면 오류 발생
    if any(i not in by_id for i in ids):
        raise ValueError("선택한 ID가 없습니다")

    # [5단계] 선택한 ID에 해당하는 지식만 추출
    # 예: ids=[102] → 102번 지식만 chosen에 저장
    chosen = [by_id[i] for i in ids]

    # [6단계] 선택한 지식을 JSON 문자열로 변환
    # ensure_ascii=False: 한글을 유니코드 코드 형태로 바꾸지 않고 그대로 유지
    context_json = json.dumps(chosen, ensure_ascii=False)

    # [7단계] 선택한 지식과 JSON 문자열 반환
    return chosen, context_json


def calculate_budget():
    # [함수 역할]
    # AI가 한 번에 처리할 수 있다고 가정한 전체 token 공간에서
    # 지시문·질문·예시·답변 등에 사용할 공간을 제외하고,
    # 참고 지식(Context)에 사용할 수 있는 공간을 계산하는 함수
    #
    # 입력: 코드에 미리 정해 놓은 가상의 token 예산
    # 출력: 항목별 예산과 Context에 사용할 수 있는 token 수

    # [1단계] 전체 token 공간과 용도별 사용량 설정
    # token: AI가 글을 처리할 때 사용하는 텍스트 단위
    parts = {
        "window": 8192,                  # 전체 사용 가능한 token 수
        "instructions_question": 800,    # 지시문과 질문에 사용할 공간
        "examples": 600,                 # Few-shot 예시에 사용할 공간
        "output_reserve": 1600,          # AI 답변을 위해 남겨 둘 공간
        "margin": 256                    # 예상치 못한 사용량에 대비한 여유 공간
    }

    # [2단계] 참고 지식에 사용할 수 있는 token 수 계산
    # 전체 공간 - 다른 용도로 사용할 공간
    available = parts["window"] - parts["instructions_question"] - parts["examples"] - parts["output_reserve"] - parts["margin"]

    # [3단계] 예산 계산 결과를 딕셔너리로 정리
    result = {
        "source": "hypothetical_budget_not_model_measurement",  # 실제 측정이 아닌 가상 예산
        "parts": parts,                                         # 항목별 token 예산
        "available_for_context": available                      # Context에 사용 가능한 token 수
    }

    # [4단계] 계산 결과가 예상값 4936과 일치하는지 확인
    # 일치하지 않으면 오류 발생
    assert available == 4936

    # [5단계] 예산 계산 결과 반환
    return result


def main():
    # [함수 역할]
    # 전체 지식에서 필요한 Context를 선택하고 token 예산을 계산한 뒤,
    # 선택한 지식과 선택·제외 기록을 파일로 저장하는 함수

    # [1단계] 전체 지식 데이터 불러오기
    # knowledge_v1.json 파일을 읽어 지식 목록을 rows에 저장
    rows = read_json(DATA / "knowledge_v1.json")

    # [2단계] 102번 지식만 선택하고 JSON 문자열로 변환
    # select_context()가 반환하는 두 결과를 각각 저장
    chosen, context_json = select_context(rows, [102])

    # [3단계] Context에 사용할 수 있는 token 예산 계산
    budget = calculate_budget()

    # [4단계] 실제 선택된 지식의 ID 목록 만들기
    # 예: chosen에 102번 지식만 있다면 included_ids=[102]
    included_ids = [r["id"] for r in chosen]

    # [5단계] 선택되지 않은 지식의 ID와 제외 이유 기록
    # 예: 전체 ID가 101, 102, 103이라면 101과 103을 제외 목록에 저장
    excluded = [
        {
            "id": r["id"],
            "reason": "이번 서버 실행 질문에서 선택하지 않음"
        }
        for r in rows if r["id"] not in included_ids
    ]

    # [6단계] Context 구성 기록 만들기
    # 사용한 지식 버전, 포함·제외 ID, token 예산을 하나로 정리
    manifest = {
        "context_version": "knowledge_v1",
        "included_ids": included_ids,
        "excluded": excluded,
        "budget": budget
    }

    # [7단계] 선택한 Context를 JSON 파일로 저장
    # AI에게 전달할 참고 지식의 원문을 보존
    write_text(HERE / "outputs" / "context.json", context_json)

    # [8단계] Context 선택 기록과 예산을 JSON 파일로 저장
    write_json(HERE / "outputs" / "context_manifest.json", manifest)

    # [9단계] 작업 완료 메시지 출력
    print("[04] Context와 가상 예산 저장 완료. token 실측·검색 결과가 아닙니다.")


# [프로그램 실행]
# 이 파일을 직접 실행했을 때만 main()을 실행
# run()은 수업에서 제공한 실행 보조 함수
if __name__ == "__main__":
    run(main)