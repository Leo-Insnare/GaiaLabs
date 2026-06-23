# Theia Clinic CCTV AI

클리닉 CCTV 영상에서 A/C/D 관측 변수를 추출하고 CRM 출력변수를 산출하는 테스트 파이프라인입니다.

실제 프로덕션 연동(CRM API 호출, DB 저장, 권한 관리)은 포함하지 않으며, 모델 추론 결과 검증과 CRM 상태 로직 확인에 집중합니다.

---

## 프로젝트 구조

```
theia_cctv_ai/
├── app.py                  # Streamlit 진입점
├── run_video.py            # CLI 진입점
├── config.py               # 경로·파라미터 설정
├── inference/
│   ├── a_detector.py       # A Group (환자 자세, 의료진 복장)
│   ├── c_detector.py       # C1 rule-based (얼굴/상단 crop → score)
│   └── d_detector.py       # D Group (의료 기기 종류)
├── pipeline/
│   ├── video_reader.py     # 프레임 샘플링
│   ├── integrator.py       # A/C/D 결과 통합
│   ├── state_machine.py    # CRM 상태 전이 + debounce
│   ├── visualizer.py       # 어노테이션 프레임 렌더링
│   ├── verification.py     # 검증 항목 기록
│   └── runner.py           # 전체 파이프라인 조율
├── utils/
│   └── io.py               # CSV/JSON/ZIP 출력
├── models/                 # .pt 파일 배치 위치 (git 제외)
├── outputs/                # 실행 결과 저장
├── requirements.txt
├── packages.txt
└── .streamlit/config.toml
```

---

## 모델 파일

기본 경로는 Colab/Google Drive 기준입니다.

```
A Group : /content/drive/MyDrive/GaiaLabs/A_group_model/a_group_yolov8n_best.pt
D Group : /content/drive/MyDrive/GaiaLabs/D_group_model/best.pt
```

로컬 실행 시 아래 세 가지 방법 중 하나를 선택합니다.

**방법 1 — 파일 직접 배치**

`models/` 폴더에 `.pt` 파일을 넣고 Streamlit UI에서 경로를 입력합니다.

**방법 2 — UI 업로드**

Streamlit에서 `Model source = Upload`를 선택하면 A/D 모델을 각각 업로드할 수 있습니다. 파일이 repo에 없을 때 가장 빠른 방법입니다.

**방법 3 — 환경변수**

```bash
export THEIA_A_MODEL_PATH=models/a_group_yolov8n_best.pt
export THEIA_D_MODEL_PATH=models/best.pt
```

> **주의** — `.pt` 파일은 용량 문제로 git에 포함하지 않습니다. `.gitignore`에 `models/` 가 등록되어 있는지 확인하세요.

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.9 이상을 권장합니다. `packages.txt`에 명시된 시스템 패키지(ffmpeg 등)가 없으면 영상 입출력에서 오류가 날 수 있습니다.

---

## 실행

### Streamlit UI

```bash
streamlit run app.py
```

브라우저에서 영상 업로드 → 모델 소스 선택 → 샘플링 FPS 조정 → Run 순서로 진행합니다.

**Streamlit Cloud 배포** — 레포를 그대로 올리고 앱 진입 파일을 `app.py`로 지정합니다. 모델 파일은 repo에 포함하지 않으므로 `Model source = Upload` 방식을 사용하세요.

### CLI

빠른 배치 처리가 필요할 때 사용합니다.

```bash
python run_video.py \
  --video sample.mp4 \
  --a-model models/a_group_yolov8n_best.pt \
  --d-model models/best.pt \
  --output-dir outputs \
  --sampling-fps 1.0
```

---

## 출력 파일

실행이 완료되면 `outputs/<run_id>/` 아래에 아래 파일이 생성됩니다.

| 파일 | 내용 |
|------|------|
| `frame_observations.csv` | 프레임별 A/C/D 추론 원시 결과 |
| `event_timeline.json` | 시간순 CRM 이벤트 목록 |
| `crm_summary.json` | 세션 전체 CRM 상태 요약 |
| `output_schema.json` | 출력 컬럼 정의 |
| `run_config.json` | 실행 시 사용된 설정값 |
| `verification_report.json` | 검증 항목 통과 여부 |
| `annotated_preview.mp4` | bbox·레이블이 합성된 확인용 영상 |
| `outputs.zip` | 위 파일 전체 압축본 |

---

## CRM 출력변수 처리 로직

### A Group

`threshold` 이상으로 검출되는 클래스가 있으면 해당 변수를 Y로 마킹합니다.

**A1** (환자 자세)
- `a1_lying_on_surface`
- `a1_sitting_on_surface`

**A2** (의료진 복장)
- `a2_medical_whitecoat`
- `a2_medical_gray_scrub`
- `a2_coordinator_black_uniform`

### C Group

**C1**은 전용 모델 없이 동작합니다. A1에서 환자 bbox가 감지된 프레임에 한해 얼굴/상단 영역을 crop한 뒤 rule score를 계산합니다. A1 bbox가 없는 프레임에서는 C1을 산출하지 않고 N 처리합니다.

### D Group

D0~D14 클래스 각각의 bbox 존재 여부를 변수로 변환합니다.

```
D0  erbium        D5  density       D10 sofwave
D1  titanium      D6  dermashine    D11 ldm
D2  thermage      D7  onda          D12 bbl
D3  potenza       D8  shurink       D13 oxygen_dome
D4  inmode        D9  ulthera       D14 led_mask
```

### 이번 버전에서 제외된 항목

아래 항목은 현재 파이프라인에 포함하지 않습니다.

```
A3 / B1 / B2 / B3
YOLO Pose
ByteTrack
Face Recognition
```

---

## CRM 상태 정의

```
S1  room_occupied                   # 실내 인원 감지
S2  simple_waiting                  # 환자 대기 상태
S3  anesthesia                      # 마취 처치 중
S4  treatment_or_procedure_candidate  # 시술 후보 (수동 확정 필요)
S5  device_present                  # 의료 기기 감지
S6  room_empty                      # 실내 비어 있음
```

S4는 자동으로 이벤트 세그먼트를 확정하지 않고 `candidate` 컬럼만 남깁니다. 실제 확정은 CRM 담당자가 수동으로 처리합니다.

---

## 검증 리포트

`verification_report.json`은 파이프라인 실행 후 자동으로 생성되며 아래 항목의 통과 여부를 기록합니다.

- A/D 모델 파일 존재 확인
- 영상 파일 open 성공 여부
- 샘플링된 프레임 수
- A 추론 컬럼 정합성
- D 추론 컬럼 정합성
- C1 결과 컬럼 정합성
- 논리 변수 컬럼 정합성
- CRM 이벤트 세그먼트 생성 여부
- CSV / JSON / 영상 출력 파일 존재 확인

---

## 결과 검토 시 확인할 것

`annotated_preview.mp4`, `frame_observations.csv`, `event_timeline.json` 세 파일을 함께 보는 것을 권장합니다.

아래는 실제 테스트에서 자주 발생했던 이슈입니다.

- **A1 오탐** — 빈 침대를 환자로 감지하는 경우
- **A2 오탐** — 검은 상의를 입은 환자·방문자를 의료진으로 분류하는 경우
- **A2 혼동** — gray scrub과 black uniform 사이의 오분류
- **D 미탐·혼동** — 유사하게 생긴 기기 간 오분류, 가려진 기기 미탐
- **C1 rule 문제** — crop 기준 미탐 또는 오탐
- **debounce 동작** — 짧은 프레임 끊김이 상태 전이로 잘못 처리되지 않는지 확인