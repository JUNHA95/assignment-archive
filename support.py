
"""
강의 공통 오프라인 계산 도구

실제 사전학습 LLM이 아니라,
Tokenizer, Cosine Similarity, Attention 등의
계산 원리를 확인하기 위한 교육용 코드
"""

from pathlib import Path
import json
import math
import re
import torch


# ============================================================
# ① DATA — 실습에 사용할 원본 문장
# ============================================================
# 목적:
# Tokenizer에 입력할 문장과 관련 정보를 준비
#
# 각 데이터의 구성:
# id       : 문장 식별자
# title    : 문장 제목
# content  : 실제 문장
# category : 문장 분류
#
# [8교시 연결]
# analysis.py에서 DATA의 content만 추출해
# ToyTokenizer의 어휘표를 만드는 데 사용
# ============================================================

DATA = [
    {
        "id": "A",
        "title": "실습 파일 보관",
        "content": "실습 파일은 개인 폴더에 보관합니다.",
        "category": "learning",
    },
    {
        "id": "B",
        "title": "서버 실행",
        "content": "프로젝트 폴더에서 서버를 실행합니다.",
        "category": "environment",
    },
    {
        "id": "C",
        "title": "질문 작성",
        "content": "오류 메시지와 시도한 작업을 함께 적습니다.",
        "category": "learning",
    },
]


# ============================================================
# ② SPECIAL — 특수 토큰 정의
# ============================================================
# [PAD]: 문장 길이를 맞추기 위해 채우는 토큰
# [UNK]: 어휘표에 없는 단어를 대신하는 토큰
# [BOS]: 문장의 시작을 나타내는 토큰
# [EOS]: 문장의 끝을 나타내는 토큰
#
# 예시:
# "서버 실행"
#
# → [BOS], 서버, 실행, [EOS]
#
# 길이를 6개로 맞추면:
# → [BOS], 서버, 실행, [EOS], [PAD], [PAD]
# ============================================================

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


# ============================================================
# ③ words() — 문장을 단어 단위로 분리
# ============================================================
# 목적:
# 문장을 토큰으로 나누기 위한 첫 번째 처리
#
# 정규표현식(re):
# 문자 패턴을 기준으로 텍스트를 찾는 기능
#
# 분리 규칙:
# [가-힣]+   → 연속된 한글
# [A-Za-z]+ → 연속된 영어
# \d+       → 연속된 숫자
# [^\w\s]   → 문장부호 등
#
# 예시:
# "서버를 실행합니다."
#
# → ["서버를", "실행합니다", "."]
#
# 주의:
# 실제 LLM에서 사용하는 서브워드 방식과 다름
# ============================================================

def words(text: str) -> list[str]:
    """교육용 규칙 기반 토큰 분리"""

    return re.findall(
        r'[가-힣]+|[A-Za-z]+|\d+|[^\w\s]',
        text
    )


# ============================================================
# ④ ToyTokenizer — 토큰을 숫자 ID로 변환
# ============================================================
# 목적:
# 문장을 토큰으로 나눈 뒤 숫자 ID로 변환
#
# [8교시 기본 A01 / 응용 A05 연결]
#
# __init__(): 어휘표 생성
# encode():   문장 하나를 토큰화
# batch():    여러 문장을 한꺼번에 토큰화
# ============================================================

class ToyTokenizer:

    # --------------------------------------------------------
    # ④-1. __init__() — 어휘표 만들기
    # --------------------------------------------------------
    # corpus:
    # Tokenizer가 참고할 문장 목록
    #
    # 처리:
    # 1. 모든 문장을 words()로 분리
    # 2. 중복 단어 제거
    # 3. 알파벳순 등 파이썬 문자열 정렬 순서로 정렬
    # 4. SPECIAL 토큰을 앞에 추가
    # 5. 각 토큰에 숫자 ID 부여
    #
    # 예시:
    # SPECIAL = [PAD, UNK, BOS, EOS]
    # 단어 = ["서버", "실행"]
    #
    # vocab:
    # PAD=0, UNK=1, BOS=2, EOS=3,
    # 서버=4, 실행=5
    #
    # reverse:
    # 숫자 ID에서 토큰을 찾기 위한 역방향 표
    # --------------------------------------------------------

    def __init__(self, corpus):

        # 전체 문장을 토큰으로 분리하고 중복 제거
        wordset = sorted({
            word
            for line in corpus
            for word in words(line)
        })

        # 토큰 → 숫자 ID
        self.vocab = {
            tok: i
            for i, tok in enumerate(SPECIAL + wordset)
        }

        # 숫자 ID → 토큰
        self.reverse = {
            i: t
            for t, i in self.vocab.items()
        }


    # --------------------------------------------------------
    # ④-2. encode() — 문장 하나를 Token ID로 변환
    # --------------------------------------------------------
    # 목적:
    # 입력 문장에 특수 토큰을 추가하고,
    # 길이를 조절한 뒤 숫자 ID로 변환
    #
    # 처리 순서:
    #
    # 1. words()로 문장 분리
    # 2. BOS/EOS 추가
    # 3. max_length를 넘으면 잘라내기
    # 4. padding_to가 지정되면 PAD 추가
    # 5. 각 토큰을 숫자 ID로 변환
    # 6. Attention Mask 생성
    #
    # 여기서 Attention Mask:
    # 실제 토큰 = 1
    # PAD 토큰 = 0
    #
    # 주의:
    # 이 Attention Mask는 Padding 구분용이며,
    # 미래 토큰을 차단하는 Causal Mask와 다름
    # --------------------------------------------------------

    def encode(self, text: str, max_length=None, padding_to=None):

        # 1. 입력 문장을 토큰으로 분리
        raw = words(text)

        # 2. 문장 시작과 끝에 특수 토큰 추가
        tokens = ["[BOS]"] + raw + ["[EOS]"]

        # 잘림 여부 기록
        truncated = False

        # 3. 최대 길이가 지정된 경우
        if max_length is not None:

            # BOS와 EOS를 포함하려면 최소 2칸 필요
            if max_length < 2:
                raise ValueError(
                    "BOS/EOS를 포함하려면 max_length>=2"
                )

            # 최대 길이를 초과하면 문장을 잘라냄
            if len(tokens) > max_length:

                # 마지막 칸은 EOS를 위해 남겨둠
                tokens = tokens[:max_length - 1] + ["[EOS]"]

                truncated = True

        # PAD를 추가하기 전의 실제 유효 길이
        valid = len(tokens)

        # 4. Padding 길이가 지정된 경우
        if padding_to is not None:

            # 실제 토큰보다 짧게 Padding할 수 없음
            if padding_to < valid:
                raise ValueError(
                    "padding_to must be >= valid length"
                )

            # 부족한 길이만큼 PAD 추가
            tokens = tokens + ["[PAD]"] * (padding_to - valid)

        # 5. 각 토큰을 숫자 ID로 변환
        # 어휘표에 없으면 UNK의 ID 사용
        ids = [
            self.vocab.get(t, self.vocab["[UNK]"])
            for t in tokens
        ]

        # 6. Padding 구분용 Attention Mask 생성
        # PAD는 0, 나머지는 1
        mask = [
            0 if t == "[PAD]" else 1
            for t in tokens
        ]

        # 처리 결과를 하나의 딕셔너리로 반환
        return {
            "tokens": tokens,
            "input_ids": ids,
            "attention_mask": mask,
            "valid_length": valid,
            "truncated": truncated,
            "original_length": len(raw) + 2,
        }


    # --------------------------------------------------------
    # ④-3. batch() — 여러 문장 한꺼번에 처리
    # --------------------------------------------------------
    # [8교시 응용 A05 연결]
    #
    # 기본:
    # encode()로 문장 하나 처리
    #
    # 응용:
    # batch()로 여러 문장을 처리
    #
    # 예시:
    # 문장 A: 토큰 4개
    # 문장 B: 토큰 6개
    #
    # 두 문장을 함께 처리하기 위해
    # 길이가 짧은 A에 PAD 2개 추가
    #
    # 결과:
    # A = [토큰, 토큰, 토큰, 토큰, PAD, PAD]
    # B = [토큰, 토큰, 토큰, 토큰, 토큰, 토큰]
    #
    # 반환값:
    # 각 문장의 encode() 결과를 담은 목록
    # --------------------------------------------------------

    def batch(self, texts, max_length=None):

        # 각 문장을 먼저 토큰화하여 길이 확인
        items = [
            self.encode(t, max_length=max_length)
            for t in texts
        ]

        # Batch에서 가장 긴 문장의 토큰 길이
        max_size = max(
            len(i["input_ids"])
            for i in items
        )

        # 모든 문장을 가장 긴 길이에 맞춰 Padding
        return [
            self.encode(
                t,
                max_length=max_length,
                padding_to=max_size,
            )
            for t in texts
        ]


# ============================================================
# ⑤ context_budget() — Context Window 용량 계산
# ============================================================
# 목적:
# 모델이 한 번에 처리할 수 있는 토큰 수를
# 초과하지 않는지 확인
#
# Context Window:
# 모델이 한 번의 처리에서 사용할 수 있는
# 입력과 출력 토큰의 전체 용량
#
# 예시:
# 전체 용량 = 1000
# 입력 사용 = 600
# 출력 예약 = 200
#
# 남은 용량:
# 1000 - 600 - 200 = 200
#
# fits:
# 전체 용량 안에 들어가면 True
# 초과하면 False
#
# 8교시 analysis.py에서는 직접 사용하지 않음
# ============================================================

def context_budget(window, parts: dict, output_reserved: int):

    # 입력의 각 부분에서 사용한 토큰 수 합계
    used = sum(parts.values())

    return {
        "window": window,
        "input_used": used,
        "output_reserved": output_reserved,
        "remaining": window - used - output_reserved,
        "fits": used + output_reserved <= window,
    }


# ============================================================
# ⑥ cosine() — 코사인 유사도 계산
# ============================================================
# [8교시 기본 A02 연결]
#
# 목적:
# 두 벡터가 얼마나 비슷한 방향인지 계산
#
# 예시:
# A = [1, 0]
# B = [0.8, 0.2]
#
# 결과:
# cosine(A, B) ≈ 0.9701
#
# 계산:
# 두 벡터의 내적을
# 각 벡터 길이의 곱으로 나눈 값
#
# PyTorch의 cosine_similarity()가
# 해당 계산을 수행
# ============================================================

def cosine(a, b):

    # 입력 데이터를 PyTorch Tensor로 변환
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)

    # 두 벡터의 코사인 유사도 계산
    # dim=0: 1차원 벡터의 요소 방향으로 계산
    return float(
        torch.nn.functional.cosine_similarity(
            a, b, dim=0
        )
    )


# ============================================================
# ⑦ attention() — Scaled Dot-Product Attention
# ============================================================
# [8교시 기본 A03 / 응용 A06 연결]
#
# 목적:
# Q, K, V를 이용해 토큰 사이의
# Attention 가중치와 출력 벡터 계산
#
# 전체 계산 순서:
#
# Q, K
#   ↓
# Q × K.T
#   ↓
# sqrt(d_k)로 나누기
#   ↓
# Causal Mask 적용(선택)
#   ↓
# Softmax
#   ↓
# Attention Weights
#   ↓
# Weights × V
#   ↓
# Attention Output
#
# d_k:
# Key 벡터의 차원 수
#
# sqrt(d_k)로 나누는 이유:
# 벡터 차원이 커질 때 점수가 지나치게 커져
# Softmax 확률이 한쪽에 과도하게 몰리는 현상을 완화
#
# 참고:
# 여기서는 Q와 K의 차원이 같으므로
# q.shape[-1]을 사용해도 됨
# ============================================================

def attention(q, k, v, causal=True):

    # 1. 입력을 Tensor로 변환
    q = torch.as_tensor(q, dtype=torch.float32)
    k = torch.as_tensor(k, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)

    # 2. Q와 K의 내적으로 Attention 점수 계산
    # K.T: K 행렬의 행과 열을 바꿈
    # sqrt(차원 수)로 나누어 점수 크기 조절
    scores = q @ k.T / math.sqrt(q.shape[-1])

    # 3. 미래 토큰 위치를 표시하는 Causal Mask 생성
    #
    # 3개 토큰 예시:
    # False True  True
    # False False True
    # False False False
    #
    # True = 차단할 위치
    mask = torch.triu(
        torch.ones_like(scores, dtype=torch.bool),
        diagonal=1,
    )

    # 4. Causal Mask 적용 후 Softmax
    #
    # causal=True:
    # 미래 토큰의 점수를 -inf로 변경
    # → Softmax 결과가 0이 됨
    #
    # causal=False:
    # Mask 없이 모든 토큰 점수를 사용
    #
    # Softmax:
    # 점수를 각 토큰에 대한 Attention 확률로 변환
    weights = torch.softmax(
        scores.masked_fill(mask, float("-inf"))
        if causal else scores,
        dim=-1,
    )

    # 5. Attention 출력 계산
    #
    # weights @ v:
    # 각 토큰의 Value 벡터를
    # Attention 가중치에 따라 합산
    #
    # 반환값:
    # scores  : Softmax 이전 점수
    # weights : Attention 가중치
    # output  : 가중치를 적용한 Value 벡터
    # mask    : 미래 토큰 차단 위치
    return scores, weights, weights @ v, mask


# ============================================================
# ⑧ write_json() — JSON 파일 저장
# ============================================================
# [8교시 기본 A10 연결]
#
# 목적:
# 계산 결과를 프로그램이 다시 읽기 쉬운
# JSON 형식으로 저장
#
# ensure_ascii=False:
# 한글을 읽을 수 있는 문자 그대로 저장
#
# indent=2:
# 들여쓰기 2칸으로 보기 쉽게 정리
#
# mkdir():
# 저장 폴더가 없으면 자동 생성
# ============================================================

def write_json(path, data):

    # 저장 경로를 Path 객체로 변환
    p = Path(path)

    # 상위 폴더가 없으면 생성
    p.parent.mkdir(exist_ok=True, parents=True)

    # 데이터를 JSON 문자열로 변환하여 저장
    p.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# ⑨ write_md() — Markdown 파일 저장
# ============================================================
# [8교시 기본 A09 연결]
#
# 목적:
# 계산 결과를 사람이 읽기 쉬운
# Markdown 문서로 저장
#
# Markdown:
# 제목, 목록, 코드 블록 등을 표현할 수 있는
# 텍스트 문서 형식
# ============================================================

def write_md(path, text):

    # 저장 경로 지정
    p = Path(path)

    # 상위 폴더가 없으면 생성
    p.parent.mkdir(exist_ok=True, parents=True)

    # Markdown 문자열 저장
    p.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# ⑩ read_json() — JSON 파일 불러오기
# ============================================================
# 목적:
# 저장된 JSON 파일을 다시 읽어
# 파이썬 데이터로 변환
#
# write_json():
# 파이썬 데이터 → JSON 파일
#
# read_json():
# JSON 파일 → 파이썬 데이터
#
# 8교시 analysis.py에서는 직접 사용하지 않음
# ============================================================

def read_json(path):

    # 파일을 UTF-8로 읽은 뒤
    # JSON 문자열을 파이썬 데이터로 변환
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )
