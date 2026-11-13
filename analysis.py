
from pathlib import Path
import sys

# ============================================================
# [실행 준비] 파일 경로와 공통 기능 불러오기
# ============================================================
# HERE: 현재 analysis.py가 위치한 폴더
# LEVEL_ROOT: HERE의 상위 폴더
#
# sys.path에 LEVEL_ROOT를 추가하는 이유:
# 어느 작업 폴더에서 실행하더라도
# common/support.py에 정의된 기능을 가져오기 위해서
# ============================================================

HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))

import math
import torch
import torch.nn.functional as F

from common.support import (
    DATA,
    ToyTokenizer,
    attention,
    cosine,
    write_json,
    write_md,
)

# 난수 사용 시 동일한 결과가 나오도록 설정
torch.manual_seed(42)

# CPU 연산에 사용할 PyTorch 스레드 수 제한
torch.set_num_threads(1)


# ============================================================
# [8교시 심화] 일일프로젝트
# 문장 하나가 LLM 출력이 되기까지
# ============================================================
#
# 목적:
# 앞선 교시에서 배운 개념을 하나의 실습에서 확인하고
# 결과를 보고서로 정리하기
#
# 전체 흐름(개념):
# 입력 문장
#   ↓
# Tokenizer → Token ID
#   ↓
# Embedding → Vector
#   ↓
# Self-Attention + Causal Mask
#   ↓
# Logits → Softmax
#   ↓
# 다음 Token 선택
#
# 주의:
# 실제로 위 계산 결과들이 차례로 전달되는
# 하나의 학습된 LLM을 구현한 것은 아님
#
# 각 단계의 원리를 독립적인 교육용 계산으로 확인하고
# 그 결과를 하나의 보고서에 통합하는 실습
# ============================================================

def run_analysis():

    # DATA의 content(문장)만 추출
    texts = [d["content"] for d in DATA]

    # 문장들로 교육용 Tokenizer 생성
    tokenizer = ToyTokenizer(texts)


    # ========================================================
    # [기본 ①] Tokenizer — 문장을 Token ID로 변환
    # ========================================================
    # 목적:
    # 사람이 작성한 문장을 컴퓨터가 처리할 수 있는
    # 숫자 ID로 바꾸는 과정 이해
    #
    # 처리:
    # 1. DATA에서 첫 번째 문장 선택
    # 2. ToyTokenizer로 토큰화
    # 3. 최대 길이 12 이내인지 검증
    #
    # 예시(개념):
    # "서버를 실행합니다"
    # → ["서버를", "실행합니다"]
    # → [15, 28]
    #
    # 실제 출력 ID는 support.py의
    # ToyTokenizer 구현과 어휘표에 따라 결정됨
    #
    # valid_length:
    # Special Token을 포함한 유효 토큰 개수
    #
    # 주의:
    # 실제 사전학습 LLM의 Tokenizer가 아닌
    # 규칙 기반 교육용 Tokenizer
    # ========================================================

    # BEGIN D20-08-A01

    # 첫 번째 문장을 입력으로 선택
    input_sentence = DATA[0]["content"]

    # 문장을 Token ID로 변환
    token_report = tokenizer.encode(
        input_sentence,
        max_length=12,
    )

    # 유효 토큰 길이가 최대 12 이하인지 확인
    assert token_report["valid_length"] <= 12

    # END D20-08-A01


    # ========================================================
    # [기본 ②] Vector + Cosine Similarity
    # ========================================================
    # 목적:
    # 벡터의 방향을 비교하여 유사도를 계산하는 원리 이해
    #
    # 실제 LLM의 Embedding:
    # 학습된 벡터 표현을 사용
    #
    # 이번 실습:
    # 사람이 직접 정한 2차원 벡터 3개 사용
    #
    # A = [1.0, 0.0]
    # B = [0.8, 0.2]
    # C = [-1.0, 0.0]
    #
    # 코사인 유사도:
    #  1에 가까움 → 방향이 비슷함
    #  0에 가까움 → 방향이 직각에 가까움
    # -1에 가까움 → 방향이 반대임
    #
    # 계산 결과:
    # A-B ≈ 0.9701
    # A-C = -1.0000
    #
    # 따라서 A와 B의 방향이 더 비슷함
    #
    # 주의:
    # 실제 문장의 의미를 분석한 결과가 아니라
    # 임의로 지정한 벡터의 유사도 계산임
    # ========================================================

    # BEGIN D20-08-A02

    # 교육용 벡터 3개
    coords = {
        "A": [1., 0.],
        "B": [.8, .2],
        "C": [-1., 0.],
    }

    # A를 기준으로 B, C와 코사인 유사도 계산
    cosine_table = {
        f"A_{k}": cosine(coords["A"], v)
        for k, v in coords.items()
        if k != "A"
    }

    # END D20-08-A02


    # ========================================================
    # [기본 ③] Self-Attention + Causal Mask
    # ========================================================
    # 목적:
    # 토큰들이 서로의 정보를 참고하는 과정과
    # 미래 토큰을 차단하는 원리 이해
    #
    # Self-Attention:
    # 각 토큰이 다른 토큰의 정보를 얼마나 참고할지 계산
    #
    # Q(Query): 어떤 정보를 찾는가?
    # K(Key):   어떤 정보를 가지고 있는가?
    # V(Value): 실제로 가져올 정보는 무엇인가?
    #
    # 이번 실습:
    # Q, K, V에 같은 숫자 벡터 사용
    #
    # Causal Mask:
    # 현재 위치보다 뒤에 있는 토큰은 참고하지 못하게 차단
    #
    # 토큰 3개라면:
    #
    #          1번  2번  3번
    # 1번       O    X    X
    # 2번       O    O    X
    # 3번       O    O    O
    #
    # O: 참고 가능
    # X: 미래 토큰이므로 차단
    #
    # 검증:
    # 1. 각 행의 Attention 가중치 합계 = 1
    # 2. 차단된 위치의 가중치 = 0
    #
    # 주의:
    # 실제 자연어를 학습한 Attention이 아니라
    # 교육용 숫자로 계산한 결과임
    # ========================================================

    # BEGIN D20-08-A03

    # 토큰 3개의 교육용 벡터
    q = torch.tensor([
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ])

    # Self-Attention 계산
    # causal=True: 미래 토큰 참조 차단
    scores, weights, attn_out, mask = attention(
        q, q, q, causal=True
    )

    # 각 행의 Attention 가중치 합계가 1인지 검증
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(3),
    )

    # Mask로 차단된 위치의 가중치가 0인지 검증
    assert torch.all(weights[mask] == 0)

    # END D20-08-A03


    # ========================================================
    # [기본 ④] Logits → Softmax → Greedy
    # ========================================================
    # 목적:
    # 다음 토큰 후보의 점수를 확률로 바꾸고
    # 가장 확률이 높은 후보를 선택하기
    #
    # 6교시에서 학습한 토큰 선택 과정
    #
    # Logits = [2, 1, 0]
    #
    # Softmax 적용:
    # 출력A ≈ 66.5%
    # 출력B ≈ 24.5%
    # 출력C ≈  9.0%
    #
    # Greedy:
    # 가장 확률이 높은 출력A 선택
    #
    # 이번 코드에서는 Temperature=1인 것과
    # 같은 확률 분포를 사용
    #
    # Top-k, Top-p, Sampling은 사용하지 않음
    #
    # 주의:
    # 앞에서 계산한 Attention 결과를 사용하지 않고
    # 별도로 지정한 Logits [2, 1, 0]을 사용
    # ========================================================

    # BEGIN D20-08-A04

    # 다음 토큰 후보의 점수
    logits = torch.tensor([2., 1., 0.])

    # Logits를 확률로 변환
    probs = torch.softmax(logits, dim=0)

    # 가장 높은 확률의 후보 인덱스 선택
    selected = int(probs.argmax().item())

    # 선택된 후보를 이름으로 변환
    next_token = [
        "출력A",
        "출력B",
        "출력C",
    ][selected]

    # END D20-08-A04


    # ========================================================
    # [응용 ①] 여러 문장 Batch Tokenization
    # ========================================================
    # 기본과의 차이:
    #
    # [기본]
    # DATA의 첫 번째 문장 하나를 토큰화
    #
    # [응용]
    # 새로운 문장 3개를 한꺼번에 토큰화하고
    # 각 문장의 유효 토큰 길이를 비교
    #
    # Batch:
    # 여러 입력을 하나로 묶어 처리하는 방식
    #
    # 문장마다 토큰 수가 다를 수 있으므로
    # 최대 길이를 맞추는 처리가 필요할 수 있음
    #
    # 여기서는 max_length=12로 처리
    #
    # 결과:
    # custom_lengths에 문장별 유효 길이 저장
    #
    # 주의:
    # 실제 토큰 분리 규칙과 Padding 방식은
    # support.py의 ToyTokenizer 구현을 따라감
    # ========================================================

    # BEGIN D20-08-A05

    # 새롭게 준비한 문장 3개
    custom_texts = [
        "실습 기록은 개인 폴더에 보관합니다.",
        "서버를 프로젝트 폴더에서 실행합니다.",
        "오류 메시지와 실행 명령을 같이 적습니다.",
    ]

    # 새 문장들로 교육용 어휘표 생성
    custom_tokenizer = ToyTokenizer(custom_texts)

    # 문장 3개를 Batch로 토큰화
    custom_batch = custom_tokenizer.batch(
        custom_texts,
        max_length=12,
    )

    # 각 문장의 유효 토큰 길이 추출
    custom_lengths = [
        row["valid_length"]
        for row in custom_batch
    ]

    # END D20-08-A05


    # ========================================================
    # [응용 ②] Causal Mask 적용 전후 비교
    # ========================================================
    # 기본과의 차이:
    #
    # [기본]
    # Causal Mask를 적용하고
    # 차단 위치의 가중치가 0인지 검증
    #
    # [응용]
    # Mask를 적용하지 않은 Attention도 계산하여
    # 두 결과가 얼마나 다른지 수치로 비교
    #
    # Mask 적용:
    # 현재 위치보다 뒤의 토큰은 참고할 수 없음
    #
    # Mask 미적용:
    # 모든 토큰이 서로를 참고할 수 있음
    #
    # 비교 방법:
    # 1. 두 Attention 가중치를 뺌
    # 2. 차이의 절댓값을 계산
    # 3. 모든 값을 합산
    #
    # mask_diff = 0:
    # 두 가중치 분포가 같음
    #
    # mask_diff > 0:
    # 두 가중치 분포에 차이가 있음
    #
    # 이 수치는 Attention 가중치의 차이이며
    # 모델 정확도나 성능 점수가 아님
    # ========================================================

    # BEGIN D20-08-A06

    # Causal Mask를 사용하지 않은 Attention 계산
    _, plain_w, _, _ = attention(
        q, q, q, causal=False
    )

    # Mask 적용 전후 가중치 차이의 절댓값 합산
    mask_diff = float(
        torch.abs(weights - plain_w).sum()
    )

    # END D20-08-A06


    # ========================================================
    # [심화 ①] 실제 서비스 구성 요소와 연결
    # ========================================================
    # 기본·응용과의 차이:
    #
    # [기본·응용]
    # FastAPI, PostgreSQL, LLM, RAG, Agent의 역할을
    # 간단한 문구로만 기록
    #
    # [심화]
    # 각 기술이 실제 서비스에서 담당하는 역할을
    # 조금 더 구체적으로 정리
    #
    # 예시 흐름:
    #
    # 사용자 질문
    #     ↓
    # FastAPI: 요청 수신 및 입력 검증
    #     ↓
    # RAG: 관련 지식 검색
    #     ↓
    # LLM: 질문과 검색 근거를 참고해 답변 생성
    #     ↓
    # FastAPI: 생성된 답변 전달
    #
    # PostgreSQL:
    # RAG가 검색할 지식 원본 등을 저장할 수 있음
    #
    # Agent:
    # 필요에 따라 검색이나 다른 도구를 선택하고
    # 여러 작업을 단계적으로 실행할 수 있음
    #
    # 주의:
    # 위 흐름은 가능한 서비스 구성 예시임
    # 이번 코드에서 실제로 연결하거나 실행하지 않음
    # ========================================================

    # BEGIN D20-08-A07

    platform_map = {
        "FastAPI":
            "HTTP 입력·검증·응답; 지식 관리 엔드포인트",

        "PostgreSQL":
            "지식 원본·분류·업데이트 기록(후속 설계)",

        "LLM":
            "입력 Token에 조건부로 다음 Token 분포 생성",

        "Retriever_RAG":
            "관련 근거 검색·출처 제공(11/20 이후 구현)",

        "Agent":
            "도구 선택·단계 실행, 권한 확인 필요(후속 구현)",
    }

    # END D20-08-A07


    # ========================================================
    # [심화 ②] 교육용 실습의 한계 구체화
    # ========================================================
    # 기본·응용과의 차이:
    #
    # [기본·응용]
    # "교육용 수치 예제는 실제 학습된 LLM과 다르다"
    # 라는 한 문장만 기록
    #
    # [심화]
    # 어떤 부분이 실제 LLM과 다른지
    # 기능별로 구체적으로 구분
    #
    # Tokenizer:
    # 실제 사전학습 모델의 Tokenizer가 아님
    #
    # Vector:
    # 학습된 Embedding이 아닌 임의 좌표
    #
    # Attention:
    # 학습된 Attention Head가 아닌 숫자 계산
    #
    # Logits:
    # 실제 모델이 예측한 점수가 아닌 임의 수치
    #
    # RAG / Agent:
    # 아직 구현되지 않았음
    #
    # 목적:
    # 계산 원리를 이해한 것과
    # 실제 LLM 서비스를 구현한 것을 구분하기
    # ========================================================

    # BEGIN D20-08-A08

    limitations = [
        "Tokenizer는 실제 사전학습 모델의 서브워드 Tokenizer가 아니다",
        "벡터는 임의 교육용 좌표로 문장 의미 검색 성능이 없다",
        "Attention은 자연어 모델의 학습된 Head가 아니라 장난감 Score 예제이다",
        "Logit은 예시 세 수치이며 실제 모델 vocabulary 분포가 아니다",
        "RAG·Agent 동작을 구현한 것이 아니며 이후 과목에서 연결한다",
    ]

    # END D20-08-A08


    # ========================================================
    # [기본 → 심화] Markdown 보고서 저장
    # ========================================================
    # [기본에서 구현]
    # 계산 결과를 Markdown 문서로 저장
    #
    # [응용에서도 유지]
    # 보고서 생성 방식은 기본과 동일
    #
    # [심화에서 유지]
    # 보고서 저장 코드는 동일하며,
    # 앞에서 계산한 결과를 모아 기록
    #
    # 파일:
    # outputs/one_page_summary.md
    #
    # 포함 내용:
    # - 입력 문장
    # - Token ID와 길이
    # - Cosine Similarity
    # - Attention 가중치 행별 합계
    # - 선택된 다음 Token
    # - 개념적인 전체 흐름도
    #
    # Mermaid:
    # 텍스트로 흐름도를 작성하는 문법
    #
    # 주의:
    # 흐름도는 실제로 연결된 실행 결과가 아니라
    # LLM 처리 과정을 이해하기 위한 개념도
    # ========================================================

    # BEGIN D20-08-A09

    lines = [
        "# 일일프로젝트: 문장 하나가 LLM 출력이 되기까지",
        "",
        f"입력: {input_sentence}",
        "",
        f"- Token 및 Special Token 포함 입력 길이: {token_report['valid_length']}",
        f"- Token IDs: {token_report['input_ids']}",
        f"- 교육용 Cosine A-B / A-C: {cosine_table['A_B']:.4f} / {cosine_table['A_C']:.4f}",
        f"- Causal Mask Attention Row Sum: {weights.sum(dim=-1).tolist()}",
        f"- 다음 Token 예시 선택: {next_token}",
        "",
        "```mermaid",
        "flowchart LR",
        "  U[입력 문장] --> T[Token/ID] --> E[Embedding] --> A[Attention/Mask] --> L[Logit/Softmax] --> O[다음 Token]",
        "  R[후속 Retriever] -. 검색 근거 .-> T",
        "  DB[(원본 PostgreSQL)] -. 원본/메타데이터 .-> R",
        "  F[FastAPI] -. 입력/응답 .-> U",
        "```",
        "",
        "주의: 그림은 개념적 위치 표시이며 오늘 실제 RAG·Agent 서비스를 실행한 것이 아닙니다.",
    ]

    # 문자열 목록을 줄바꿈으로 연결해 Markdown 저장
    write_md(
        HERE / "outputs" / "one_page_summary.md",
        "\n".join(lines) + "\n",
    )

    # END D20-08-A09


    # ========================================================
    # [기본 → 응용 → 심화] JSON 결과 저장
    # ========================================================
    # 목적:
    # 실습에서 계산한 결과를 하나의 JSON으로 저장
    #
    # [기본]
    # 토큰화, 코사인 유사도, Attention,
    # Logits 및 토큰 선택 결과 저장
    #
    # [응용]
    # custom_token_lengths:
    # 여러 문장의 토큰 길이 결과 추가
    #
    # masked_unmasked_l1:
    # Mask 적용 전후 가중치 차이 추가
    #
    # [심화]
    # platform:
    # 서비스 구성 요소별 역할을 구체적으로 기록
    #
    # limitations:
    # 실제 LLM과의 차이를 기능별로 구체화
    #
    # Markdown과 JSON의 차이:
    #
    # Markdown → 사람이 읽기 쉬운 보고서
    # JSON     → 프로그램이 읽기 쉬운 구조화 데이터
    #
    # 저장 파일:
    # outputs/daily_project.json
    # ========================================================

    # BEGIN D20-08-A10

    write_json(
        HERE / "outputs" / "daily_project.json",
        {
            # 원본 실습 데이터
            "source_data": DATA,

            # Tokenizer 결과
            "token": token_report,

            # 코사인 유사도 결과
            "cosine": cosine_table,

            # Attention 점수 행렬 크기
            "scores_shape": list(scores.shape),

            # Attention 가중치
            "attention_weights": weights.tolist(),

            # 각 행의 가중치 합계
            "row_sums": weights.sum(dim=-1).tolist(),

            # 미래 토큰 차단 위치
            "causal_mask": mask.tolist(),

            # 다음 토큰 후보 점수
            "logits": logits.tolist(),

            # Softmax 확률
            "probabilities": probs.tolist(),

            # Greedy로 선택한 토큰
            "next_token_demo": next_token,

            # [응용] 여러 문장의 유효 토큰 길이
            "custom_token_lengths": custom_lengths,

            # [응용] Mask 적용 전후 가중치 차이
            "masked_unmasked_l1": mask_diff,

            # [심화] 서비스 구성 요소의 역할
            "platform": platform_map,

            # [심화] 교육용 실습의 한계
            "limitations": limitations,

            # 전체 실습의 범위
            "scope":
                "교육용 독립 수치 계산; 실제 LLM/RAG/Agent 아님",
        },
    )

    print(
        "[08] daily_project.json + one_page_summary.md 생성 완료"
    )

    # END D20-08-A10
