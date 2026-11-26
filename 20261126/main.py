"""08 · FastAPI → Agent → Tool → 응답 · 기본필수 · 실제 11/20 report_agent_lab 기반 실습."""
from pathlib import Path
import sys,argparse

# 공통 실습 모듈(coursekit)을 불러올 수 있도록 경로 설정
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import tasks


def main():

    # --------------------------------------------------
    # [1단계] 실행 방식 선택
    # --------------------------------------------------
    # 터미널에서 입력한 옵션에 따라 실행 방식이 달라짐.
    #
    # ① --check: 코드 구조만 점검
    # ② --serve: FastAPI 서버 실행
    # ③ 옵션 없음: Agent를 직접 실행하여 결과 확인

    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()


    # --------------------------------------------------
    # [2단계] 코드 구조 점검 (--check)
    # --------------------------------------------------
    # tasks.py에 어떤 함수가 있는지 확인하고 실행을 종료함.
    # 실제 DB 조회, GPU 사용, LLM 호출은 하지 않음.

    if args.check:
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return

    serve=args.serve
    import starter


    # --------------------------------------------------
    # [3단계] 기본 질문 준비
    # --------------------------------------------------
    # Agent를 직접 실행할 때 사용할 질문을 미리 준비함.
    #
    # 예: RAG 근거 인용 정확도 향상 관련 보고서의 내용과
    #     63번 보고서의 인덱싱 상태를 알려줘.
    #
    # FastAPI 서버 실행 시에는 이 질문 대신
    # 사용자가 API를 통해 전달한 질문을 사용함.

    question=tasks.default_question()


    # --------------------------------------------------
    # [4단계-A] FastAPI 서버 실행 (--serve)
    # --------------------------------------------------

    if serve:

        # ① 실행 환경 점검
        # require_cuda(): GPU 사용 가능 여부 확인
        # preflight(): 기존 PostgreSQL DB의 실습 준비 상태 확인

        from coursekit.gpu import require_cuda
        from coursekit.db import preflight
        require_cuda();print('[기존 DB]',preflight())


        # ② FastAPI 앱 구성 및 서버 실행
        # make_api(): 사용자 요청을 Agent에 전달하고
        #             실행 결과를 반환할 수 있도록 API 구성
        # uvicorn.run(): 구성된 FastAPI 앱을 서버로 실행
        #
        # 서버 주소: http://127.0.0.1:8008
        # 서버 실행 후에는 사용자의 요청을 기다림.

        import uvicorn
        uvicorn.run(starter.make_api(tasks.api_response),host='127.0.0.1',port=8008)


    # --------------------------------------------------
    # [4단계-B] Agent 직접 실행 (옵션 없음)
    # --------------------------------------------------

    else:
        from coursekit.agent import run_agent

        # ① run_agent(question)
        # 기본 질문을 Agent에 전달하여 전체 실행 과정 진행
        #
        # ② api_response()
        # Agent 실행 결과를 확인하고 API 응답 형식으로 구성
        #
        # ③ print()
        # 최종 결과를 터미널에 출력
        #
        # FastAPI 서버를 실행하지 않고
        # Agent의 작동 여부를 직접 확인하는 방식임.

        print('[실제 Agent]',tasks.api_response(run_agent(question)))


if __name__=='__main__':main()