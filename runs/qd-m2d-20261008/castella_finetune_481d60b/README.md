# QD-DETR CASTELLA 미세조정, M2D-CLAP 특징, 기존 코드 481d60b (2026-10-08)

## 무엇을 하나
- 기존 코드(포크 481d60b, `legacy/task6-qd-detr`)의 QD-DETR을 CASTELLA로 미세조정한다. MS-CLAP Clotho-Moment 사전학습 → CASTELLA 미세조정(`results/qd-clotho-pretrain-20261007`)에 쓴 코드와 같다.
  - 공식 수정 두 가지(bf0e88f T2V attention mask, 35fcbaf CASTELLA 타임스탬프)는 들어 있지 않다.
  - 시작 가중치는 같은 seed의 M2D Clotho-Moment 사전학습 best checkpoint다(`results/qd-m2d-clotho-pretrain-20261008/results/clotho_pretrain_m2d_s<seed>`).
- 미세조정이 끝나면 best checkpoint로 val과 test를 추론한다. test는 seed마다 한 번만 채점한다.
- 2026-10-08 사용자 지시: 수정 코드(cbbb5be)로 한 seed 2023 미세조정(`results/qd-m2d-castella-finetune-20261008`)은 지우고, 기존 코드로 다시 돌린다.

## 사전학습 체크포인트를 그대로 쓰는 근거
- M2D 사전학습은 수정 코드 cbbb5be로 돌렸다. 하지만 Clotho-Moment 오디오는 전부 60프레임이라 오디오 패딩이 없다.
- mask 수정은 오디오 패딩 mask와 텍스트 패딩 mask의 곱으로 만든 attention mask의 배열 순서만 바꾼다. 오디오 패딩이 없으면 이 mask가 전부 False라서 두 코드의 계산이 같다.
- MS-CLAP 특징으로 확인한 결과, 수정 코드 seed 2023 사전학습 12 epoch가 기존 코드 결과와 완전히 같았다.
- 사용자 결정(2026-10-08): 사전학습은 다시 하지 않고 기존 체크포인트를 쓴다.

## 이전 폴더(cbbb5be)와의 차이
- **코드:** `legacy/task6-qd-detr`(481d60b). `start_finetune.sh`는 HEAD가 481d60b인지, `src/`와 `config.yml`에 로컬 변경이 없는지 확인한 뒤 시작한다.
- **주석:** 481d60b의 `data/castella_*_release.jsonl`에서 이전 폴더와 같은 질의 ID만 남겼다(train 2,158 / val 343 / test 1,333, 순서도 같다). 이전 폴더와 다른 줄은 두 개다.
  - train `67AE9ZKvECs_1`: `[299, 4800]`(기존 주석 그대로)
  - test `Rp4Ct_TQvAM_1`: 마지막 구간 `[298, 301]`(기존 주석 그대로)
  - 뺀 질의 47개는 `data/excluded_queries.json`에 있다. 이전 폴더와 같다.
- **설정:** 공식 `config.yml`은 두 코드에서 같다. `make_configs.sh`로 seed, results_dir, 특징 폴더, 데이터 경로만 바꿨다.
- **특징:** `features/`는 이전 폴더의 링크와 `manifest.json`을 그대로 복사했다.
- **래퍼:** `qd_resumable.py`는 이전 폴더와 docstring만 다르다. 실행 identity에 코드 해시(래퍼와 `src/*.py`)가 들어가므로 cbbb5be 실행 상태로는 재개되지 않는다.
- **컨테이너 이름:** `qd-m2d-castella-finetune-481d60b-s<seed>`

## 실행
```
./start_finetune.sh 2023        # 시작과 재개 모두 같은 명령
docker stop qd-m2d-castella-finetune-481d60b-s2023   # 중지
```
- 로그: `logs/finetune_s<seed>.log`, `logs/start.log`
- 결과: `results/castella_finetune_m2d_s<seed>/`(DONE.json, eval_val/, eval_test/)
