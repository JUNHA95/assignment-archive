
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))

import torch
import torch.nn.functional as F
from common.support import write_json

torch.manual_seed(42)
torch.set_num_threads(1)


def run_analysis():

    # ============================================================
    # [기본] 1단계. Logits → Softmax
    # ============================================================
    # 목적: AI가 계산한 점수를 실제 선택 확률로 바꾸기
    #
    # 예시: AI가 다음에 나올 토큰 후보 3개를 평가했다고 가정
    # 출력A: 2점 / 출력B: 1점 / 출력C: 0점
    #
    # Logits: AI가 각 후보에 매긴 점수 (아직 확률이 아님)
    # Softmax: 점수를 합계 100%인 확률로 변환
    #
    # 결과: A 약 66.5%, B 약 24.5%, C 약 9.0%
    # ============================================================

    logits = torch.tensor([2.0, 1.0, 0.0])
    labels = ["출력A", "출력B", "출력C"]

    # BEGIN D20-06-A01
    prob = torch.softmax(logits, dim=0)

    # 세 후보의 확률 합계가 1(100%)인지 확인
    assert torch.isclose(
        prob.sum(), torch.tensor(1.0), atol=1e-6
    )
    # END D20-06-A01


    # ============================================================
    # [기본] 2단계. Temperature — 확률 분포 조절
    # ============================================================
    # 목적: 다음 토큰을 얼마나 다양하게 선택할지 조절하기
    #
    # Temperature가 낮으면:
    #   → 확률이 높은 A에 더욱 집중
    #   → 일관된 답변을 만들기 유리
    #
    # Temperature가 높으면:
    #   → B, C가 선택될 가능성도 증가
    #   → 다양한 답변을 만들기 유리
    #
    # 예시:
    # T=0.5 → A에 확률이 크게 집중
    # T=1.0 → 원래 확률 유지
    # T=2.0 → A, B, C의 확률 차이가 줄어듦
    #
    # 주의: Temperature는 모델의 지식을 바꾸지 않음
    # ============================================================

    # BEGIN D20-06-A02
    temperature_results = {
        str(t): torch.softmax(logits / t, dim=0).tolist()
        for t in (0.5, 1.0, 2.0)
    }
    # END D20-06-A02


    # ============================================================
    # [기본] 3단계. Greedy — 가장 유력한 토큰 선택
    # ============================================================
    # 목적: 다음 토큰을 실제로 하나 선택하기
    #
    # Greedy: 확률이 가장 높은 후보를 선택하는 방법
    #
    # 예시:
    # A 66.5%, B 24.5%, C 9.0%
    # → 가장 높은 A 선택
    #
    # Autoregressive:
    # 선택한 토큰을 기존 문장 뒤에 붙이고,
    # 다음 토큰을 예측하는 과정을 반복하는 방식
    #
    # 학습(Training): 정답과 비교하며 모델의 가중치를 수정
    # 추론(Inference): 가중치를 수정하지 않고 답변 생성
    # ============================================================

    # BEGIN D20-06-A03

    # argmax: 가장 큰 확률의 위치(index)를 찾음
    selected_index = int(torch.argmax(prob).item())

    # 선택된 위치에 해당하는 토큰 이름 확인
    selected_token = labels[selected_index]

    training_vs_inference = {
        "training": (
            "정답 Token과 예측을 비교해 "
            "loss/backward/optimizer.step으로 파라미터 갱신"
        ),
        "inference": (
            "가중치를 갱신하지 않고 "
            "현재 문맥에서 다음 Token을 선택"
        ),
        "autoregressive": (
            "선택한 Token을 문맥에 추가하고 "
            "EOS/길이 제한까지 반복"
        ),
    }
    # END D20-06-A03


    # ============================================================
    # [응용] 4단계. Top-k — 후보 개수 제한
    # ============================================================
    # 목적: 가능성이 낮은 후보를 제외하기
    #
    # Top-k: 확률이 높은 상위 k개 후보만 남기는 방법
    #
    # 예시: k=2
    # A 66.5% → 유지
    # B 24.5% → 유지
    # C  9.0% → 제외
    #
    # 남은 A와 B의 확률 합계를 다시 100%로 조정
    # → A 약 73.1%, B 약 26.9%
    #
    # Greedy와 차이:
    # Greedy는 가장 높은 후보 1개를 바로 선택
    # Top-k는 선택 가능한 후보 k개를 남김
    # ============================================================

    # BEGIN D20-06-A04
    k = 2

    # 확률이 높은 상위 2개의 값과 위치 추출
    values, indices = torch.topk(prob, k)

    # 남은 두 후보의 확률 합계를 1로 다시 조정
    normalized = (values / values.sum()).tolist()

    top_k = {
        "indices": indices.tolist(),
        "labels": [labels[i] for i in indices],
        "normalized_probs": normalized,
    }
    # END D20-06-A04


    # ============================================================
    # [응용] 5단계. Sampling — 확률에 따라 무작위 선택
    # ============================================================
    # 목적: 매번 가장 높은 확률의 토큰만 선택하지 않기
    #
    # Greedy: 가장 높은 확률의 A를 항상 선택
    # Sampling: 확률을 반영해 무작위로 하나 선택
    #
    # 예시:
    # A 66.5% → 선택될 가능성이 가장 높음
    # B 24.5% → 선택될 수도 있음
    # C  9.0% → 가능성은 낮지만 선택될 수도 있음
    #
    # manual_seed(42):
    # 실습을 반복해도 같은 무작위 결과가 나오도록 설정
    #
    # 주의: 이번 코드는 Top-k 결과가 아니라
    # 전체 확률(prob)에서 Sampling을 수행함
    # ============================================================

    # BEGIN D20-06-A05
    generator = torch.Generator().manual_seed(42)

    # 확률에 따라 후보 하나를 무작위 선택
    sampled_index = int(
        torch.multinomial(
            prob, 1, generator=generator
        ).item()
    )

    sampled_token = labels[sampled_index]
    # END D20-06-A05


    # ============================================================
    # [심화] 6단계. Top-p — 누적 확률로 후보 제한
    # ============================================================
    # 목적: 상황에 따라 선택 가능한 후보 수를 유연하게 조절
    #
    # Top-k: 후보 개수를 기준으로 선택
    # Top-p: 후보들의 누적 확률을 기준으로 선택
    #
    # 예시: p=0.75 (75%)
    #
    # A 66.5% → 누적 66.5% → 유지
    # B 24.5% → 누적 91.0% → 유지
    # C  9.0% → 누적 100%  → 제외
    #
    # 75%를 처음 넘기는 B까지 포함해야 함
    #
    # Top-k와 차이:
    # Top-k는 항상 k개를 남김
    # Top-p는 확률 분포에 따라 남는 후보 수가 달라짐
    # ============================================================

    # BEGIN D20-06-A06

    # 확률이 높은 순서대로 정렬
    sorted_p, sorted_ix = torch.sort(
        prob, descending=True
    )

    # 앞에서부터 확률을 누적해서 더함
    csum = torch.cumsum(sorted_p, dim=0)

    # 누적 확률이 75%를 넘는 위치 표시
    remove = csum > 0.75

    # 75%를 처음 넘긴 후보까지는 남기도록 한 칸 이동
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    # 제거하지 않을 후보만 선택
    keep = ~remove
    kept_ix = sorted_ix[keep]
    kept_p = sorted_p[keep]

    # 남은 후보들의 확률 합계를 다시 1로 조정
    top_p = {
        "indices": kept_ix.tolist(),
        "labels": [labels[i] for i in kept_ix],
        "normalized_probs": (
            kept_p / kept_p.sum()
        ).tolist(),
    }
    # END D20-06-A06


    # ============================================================
    # [심화] 7단계. Training과 Inference의 실제 차이
    # ============================================================
    # 목적: 학습할 때와 실제 답변을 만들 때의 차이 확인
    #
    # 학습(Training):
    # ① 입력을 보고 예측
    # ② 정답과 비교해 오차(Loss) 계산
    # ③ 오차를 줄이도록 가중치 수정
    #
    # 추론(Inference):
    # ① 입력을 보고 예측
    # ② 결과를 사용
    # ③ 가중치는 수정하지 않음
    #
    # 이번 실습은 간단한 Linear 모델을 사용
    # 실제 LLM 전체를 학습하는 코드는 아님
    # ============================================================

    # BEGIN D20-06-A07

    # 입력 4개를 받아 출력 점수 3개를 만드는 간단한 모델
    model = torch.nn.Linear(4, 3, bias=False)

    # SGD: 오차를 줄이는 방향으로 가중치를 수정하는 방법
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1
    )

    # 연습용 입력과 정답
    x = torch.tensor([[1., 0., 0., 0.]])
    target = torch.tensor([1])

    # 학습 전 가중치를 복사해 보관
    old_weight = model.weight.detach().clone()

    # ----- 학습 시작 -----

    # 이전에 계산된 기울기 초기화
    optimizer.zero_grad()

    # 예측과 정답을 비교해 오차 계산
    loss = F.cross_entropy(model(x), target)

    # 오차를 줄이기 위해 각 가중치를
    # 어느 방향으로 수정해야 할지 계산
    loss.backward()

    # 계산된 방향을 이용해 실제 가중치 수정
    optimizer.step()

    # 학습 전후 가중치가 달라졌는지 확인
    changed = not torch.allclose(
        old_weight, model.weight.detach()
    )
    assert changed

    # ----- 추론 시작 -----

    # 가중치를 수정하지 않고 예측 점수만 계산
    with torch.no_grad():
        eval_scores = model(x)

    training_check = {
        "train_loss": float(loss.item()),
        "weights_changed": changed,
        "inference_scores": (
            eval_scores.squeeze(0).tolist()
        ),
    }
    # END D20-06-A07


    # ============================================================
    # [심화] 8단계. KV Cache — 이전 계산 재사용
    # ============================================================
    # 목적: LLM이 토큰을 하나씩 생성할 때 반복 계산 줄이기
    #
    # 예시: "배는 달고 맛있다"
    #
    # KV Cache가 없다면:
    # 새 토큰을 생성할 때 이전 토큰들의 K/V를 다시 계산
    #
    # KV Cache가 있다면:
    # 이전 토큰들의 Key/Value 계산 결과를 저장해 재사용
    # 새 토큰에 필요한 계산만 추가
    #
    # 장점: 반복 계산 감소 → 생성 속도 개선
    # 단점: 토큰이 많아질수록 메모리 사용량 증가
    #
    # 주의: KV Cache는 새로운 지식을 학습하거나
    # 대화 내용을 영구 기억하는 기능이 아님
    #
    # 이번 코드에서는 KV Cache를 직접 구현하지 않고
    # 개념과 특징만 JSON에 기록함
    # ============================================================

    # BEGIN D20-06-A08
    kv_note = {
        "stores": (
            "이전에 계산한 Attention Key/Value를 "
            "재사용해 반복 계산을 줄임"
        ),
        "not": (
            "영구 기억이나 사실성 검증, "
            "학습된 새 지식이 아님"
        ),
        "tradeoff": (
            "시퀀스가 길면 Cache 메모리 사용량이 증가"
        ),
    }
    # END D20-06-A08


    # ============================================================
    # [공통] 9단계. 실습 결과 저장
    # ============================================================
    # 목적: 각 단계의 계산 결과를 JSON 파일로 확인
    #
    # 저장 내용:
    # - logits: 후보별 원래 점수
    # - probs: Softmax로 계산한 확률
    # - temperature: 온도별 확률 변화
    # - greedy: 가장 높은 확률의 토큰
    # - sampling: 확률에 따라 선택한 토큰
    # - top_k: 상위 k개 후보
    # - top_p: 누적 확률 기준으로 남긴 후보
    # - training_step: 학습 전후 가중치 변경 확인
    # - kv_cache: KV Cache의 역할과 특징
    # ============================================================

    # BEGIN D20-06-A09
    write_json(
        HERE / "outputs" / "inference_report.json",
        {
            "logits": logits.tolist(),
            "probs": prob.tolist(),
            "temperature": temperature_results,
            "greedy": {
                "index": selected_index,
                "token": selected_token,
            },
            "sampling": {
                "index": sampled_index,
                "token": sampled_token,
            },
            "top_k": top_k,
            "top_p": top_p,
            "training_vs_inference": training_vs_inference,
            "training_step": training_check,
            "kv_cache": kv_note,
        },
    )

    print(
        "[06]",
        selected_token,
        "probs",
        prob.round(decimals=3).tolist(),
    )
    # END D20-06-A09
