"""미완성 TODO와 별개로 API를 열 수 있게 제공하는 고정 Starter. 수정 대상 아님."""
from coursekit.agent import run_agent


def make_api(api_response):

    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,Field


    # --------------------------------------------------
    # [사전 구성 1단계] 사용자 요청의 입력 형식 설정
    # --------------------------------------------------
    # 사용자가 API에 질문을 보낼 때 지켜야 할 조건을 설정함.
    #
    # 입력 항목: question
    # 자료형: 문자열(str)
    # 입력 조건: 최소 1자, 최대 1000자
    #
    # 경우 A: 정상 입력
    # {"question": "63번 보고서의 인덱싱 상태를 알려줘."}
    # → Agent 실행 가능
    #
    # 경우 B: 잘못된 입력
    # {"question": ""}
    # → 입력 조건을 충족하지 못하므로 422 오류 반환
    #
    # 입력값 검증은 Agent가 실행되기 전에 이루어짐.

    class Ask(BaseModel):
        question:str=Field(min_length=1,max_length=1000)


    # --------------------------------------------------
    # [사전 구성 2단계] FastAPI 앱 생성
    # --------------------------------------------------

    app=FastAPI(title='Day29 Report Agent (Local Only)')


    # --------------------------------------------------
    # [사전 구성 3단계] 사용자 요청을 접수할 API 설정
    # --------------------------------------------------
    # 사용자가 POST 방식으로 /api/agent/ask에 질문을 보내면
    # 아래 ask() 함수가 실행되도록 연결함.

    @app.post('/api/agent/ask')
    def ask(body:Ask):

        # --------------------------------------------------
        # [실행 1단계] 사용자 질문 접수 및 Agent 실행
        # --------------------------------------------------
        #
        # ① body.question
        # 입력값 검증을 통과한 사용자의 질문
        #
        # ② run_agent()
        # 사용자 질문을 전달받아 Agent의 전체 실행 과정을 진행함.
        #
        # 요청 분석 → Tool 선택 → Tool 실행 요청
        # → Tool 실행 → 실행 결과 전달 → 실행 결과 확인
        # → 최종 응답
        #
        # 예: 보고서 내용과 63번 보고서의 상태를 함께 요청하면
        # Agent는 필요에 따라 RAG Tool과 PostgreSQL Tool을 사용함.
        #
        # ③ api_response()
        # Agent 실행 완료 여부 및 출처 검증 결과를 확인하고
        # 최종 결과를 API 응답 형식으로 구성함.
        #
        # ④ return
        # 구성된 응답을 사용자에게 반환함.

        try:return api_response(run_agent(body.question))


        # --------------------------------------------------
        # [실행 2단계] 오류 발생 시 유형별 처리
        # --------------------------------------------------
        # 오류가 발생하면 정상 응답 대신
        # 오류 유형에 해당하는 HTTP 상태 코드와 메시지를 반환함.


        # 경우 ① 잘못된 입력값 (422)
        # 예: 보고서 ID로 -1이 전달되어 ValueError가 발생한 경우

        except ValueError:raise HTTPException(422,'잘못된 입력')


        # 경우 ② 접근 정책 위반 (403)
        # 예: 읽기 전용 정책에 어긋나는 작업으로
        #     PermissionError가 발생한 경우

        except PermissionError:raise HTTPException(403,'접근 정책 위반')


        # 경우 ③ Agent 실행 결과 또는 출처 검증 실패 등 (502)
        # 예: 검색을 통해 제공받지 않은 자료를 출처로 인용하여
        #     api_response()에서 RuntimeError가 발생한 경우

        except RuntimeError as exc:raise HTTPException(502,'Agent 결과/출처 검증 실패') from exc


        # 경우 ④ 그 외 실행 오류 (503)
        # 예: DB 연결 장애가 발생하고 해당 오류가
        #     위의 세 가지 오류 유형에 해당하지 않는 경우

        except Exception as exc:raise HTTPException(503,'Agent Tool 실행 실패') from exc


    # --------------------------------------------------
    # [사전 구성 4단계] 구성된 FastAPI 앱 반환
    # --------------------------------------------------
    # 반환된 앱은 main.py의 uvicorn.run()에 전달되어
    # 실제 서버로 실행됨.

    return app