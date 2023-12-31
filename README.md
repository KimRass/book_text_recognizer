# Process
- [[사법 통역의 이론과 실제]](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=105588449)라는 책을 예로 들어 진행해 보겠습니다.
## Step 1
- How to run:
    ```sh
    # Example
    python step1_parse_pdf_files.py\
        --pdf_path=".../book.pdf"\
        --save_dir="..."
    ```
- ['CRAFT' text detector의 공식 레포지토리](https://github.com/clovaai/CRAFT-pytorch)에서 pre-trained models 'craft_mlt_25k.pth', 'craft_refiner_CTW1500.pth'를 다운받아 'craft' 폴더에 넣습니다.
- 책을 pdf 파일의 형태로 준비합니다.
- pdf 파일의 각 페이지를 png 파일로 저장합니다.
- 269 페이지 이미지
    - <img src="https://i.imgur.com/HlRJnwg.jpg" width="600">
## Step 2
- How to run:
    ```sh
    # Example
    python step2_save_line_score_maps.py\
        --save_dir="..."
    ```
- 각 페이지에 대해 'CRAFT'를 사용해 Line score map을 생성하고 저장합니다.
- Line score map
    - <img src="https://i.imgur.com/4jQnSpN.png" width="600">
## Step 3
- How to run:
    ```sh
    # Example
    python step3_recognize_texts.py\
        --save_dir="..."
    ```
- Line score map의 각 Line별로 구분하는 Line segmentation map과 블록 단위로 구분하는 Block segmentation map을 생성합니다.
- Block segmentation map이 필요한 이유는 대화문이 어디서 시작하고 끝나는지 파악하기 위함입니다. Block segmentation map이 없다면 하나의 발화문 또는 비대화문이 2개 이상의 페이지에 걸쳐 연속될 때 발화자를 파악하기 어렵습니다.
- Line segmentation map
    - <img src="https://i.imgur.com/CzMb18y.png" width="600">
- Block segmentation map
    - <img src="https://i.imgur.com/H2DXlZP.png" width="600">
- CLOVA OCR API를 통해 텍스트를 인식합니다.
- CLOVA OCR 결과
    |block|line|text|
    |-|-|-|
    |1|1|기다렸다 들어와서 이제 4개월, 6개월 일한 상태라 빚이 있단 말입니다.|
    |1|2|변호사: 네, 알겠습니다. 법적으로 해결할 방법을 알아봅시다.|
    |2|3|2) 면담 Ⅱ (마약 사건)|
    |3|4|다음은 국제우편물로 마약을 반입하려다 마약류관리에관한법률위반(향정)으로|
    |3|5|불구속기소된 영국인 여성이 국선변호인과 면담하는 상황이다.|
    |4|6|변호인: 다이아나 씨 안녕하세요? 국선변호인 박승소입니다. 제가 오늘 여러 가지 공소|
    |4|7|사실과 관련해서 질문 드리고 재판 준비를 위해 오시라고 했어요. 공소장과 소|
    |4|8|환장 받아보셨죠.|
    |4|9|외국인: 예.|
    |4|10|변호인: 국적은 영국이시고, 생년월일은 기재된 대로입니까?|
    |4|11|외국인: 예|
    |4|12|변호인: 직업이 어떻게 되세요?|
    |4|13|외국인: 영어 유치원에서 강사에요. 지금 잠깐 몸이 안 좋아 쉬고 있어요.|
    |4|14|변호인: 특별히 건강이 안 좋으세요?|
    |4|15|외국인: 아니 그건 아니고 저기 그 이번 일로 좀 속을 끓이고 걱정하면서 식사도 잘 못|
    |4|16|하다 보니 좀 몸이 안 좋아져서 일하러 못 가겠어요.|
    |4|17|변호인: 너무 걱정 마세요 다이아나 씨. 기운 내서 재판을 잘 받아야지요. 그럼 차례로|
    |4|18|질문 드릴게요. 2017. 1. 29.경 국제특급우편으로 과자박스에 향정신성약품|
    |4|19|속칭 치즈 15정을 넣어서 들여온 사실이 있습니까?|
    |4|20|외국인: 예.|
    |4|21|변호인: 우편물 수령 주소는 어디로 했습니까?|
    |4|22|외국인: 제가 사는 집 주소에요.|
    |4|23|변호인: 이 우편물이 영국의 캔디 벅스가 보내준 거는 맞아요?|
    |4|24|외국인: 예.|
    |4|25|변호인: 이거 보내준 사람이 누구에요? 인터넷에서 산 거예요?|
    |4|26|외국인: 아뇨, 친구예요.|
    |5|27|7장 형사사건 통역 267|
## Step 4
- How to run:
    ```sh
    # Example
    python step4_postprocess.py\
        --save_dir="..."
    ```
- 다음과 같은 후처리 작업을 진행합니다. 대상이 되는 책에 따라 서로 다른 후처리 방법을 사용합니다.
    - 페이지 제거
    - 제목 제외
    - 문장 단위로 텍스트 병합 또는 분리
    - 화자와 발화 서로 분리
- Final result
    |page|speaker|utterance|
    |-|-|-|
    |267|변호사|기다렸다 들어와서 이제 4개월, 6개월 일한 상태라 빚이 있단 말입니다.|
    |267|변호사|네, 알겠습니다.|
    |267|변호사|법적으로 해결할 방법을 알아봅시다.|
    |267|-|다음은 국제우편물로 마약을 반입하려다 마약류관리에관한법률위반(향정)으로불구속기소된 영국인 여성이 국선변호인과 면담하는 상황이다.|
    |267|변호인|다이아나 씨 안녕하세요?|
    |267|변호인|국선변호인 박승소입니다.|
    |267|변호인|제가 오늘 여러 가지 공소사실과 관련해서 질문 드리고 재판 준비를 위해 오시라고 했어요.|
    |267|변호인|공소장과 소환장 받아보셨죠.|
    |267|외국인|예.|
    |267|변호인|국적은 영국이시고, 생년월일은 기재된 대로입니까?|
    |267|외국인|예|
    |267|변호인|직업이 어떻게 되세요?|
    |267|외국인|영어 유치원에서 강사에요.|
    |267|외국인|지금 잠깐 몸이 안 좋아 쉬고 있어요.|
    |267|변호인|특별히 건강이 안 좋으세요?|
    |267|외국인|아니 그건 아니고 저기 그 이번 일로 좀 속을 끓이고 걱정하면서 식사도 잘 못하다 보니 좀 몸이 안 좋아져서 일하러 못 가겠어요.|
    |267|변호인|너무 걱정 마세요 다이아나 씨.|
    |267|변호인|기운 내서 재판을 잘 받아야지요.|
    |267|변호인|그럼 차례로질문 드릴게요.|
    |267|변호인|2017. 1. 29.경 국제특급우편으로 과자박스에 향정신성약품속칭 치즈 15정을 넣어서 들여온 사실이 있습니까?|
    |267|외국인|예.|
    |267|변호인|우편물 수령 주소는 어디로 했습니까?|
    |267|외국인|제가 사는 집 주소에요.|
    |267|변호인|이 우편물이 영국의 캔디 벅스가 보내준 거는 맞아요?|
    |267|외국인|예.|
    |267|변호인|이거 보내준 사람이 누구에요?|
    |267|변호인|인터넷에서 산 거예요?|
    |267|외국인|아뇨, 친구예요.|
