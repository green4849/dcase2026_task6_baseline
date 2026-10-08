# QD-DETR Clotho-Moment 사전학습, M2D-CLAP 특징 (2026-10-08)

## 무엇을 하나
- 사용자 지시(2026-10-08)로 수정된 코드(포크 cbbb5be)의 QD-DETR을 Clotho-Moment에서 사전학습한다.
  - 저장된 M2D-CLAP 오디오·텍스트 특징을 입력으로 쓴다.
  - 초기 가중치부터 학습하고, MS-CLAP 기반 체크포인트에서 시작하지 않는다.
  - M2D-CLAP 추출기는 학습하지 않는다.
- seed 2023, 2024, 2025를 GPU0에서 차례로 돈다. 완성된 체크포인트는 나중에 CASTELLA 미세조정에 쓴다.
- 설정은 공식 `config_pretraining.yml` 그대로다(200 epoch, lr 1e-4, batch 32, 매 epoch Clotho val 4,918개로 평가, best val R1@0.7 체크포인트). 바꾼 것은 seed, results_dir, `a_feat_dir`, `t_feat_dir`뿐이다(`make_configs.sh`).
  - 차원은 768로 그대로다.
  - `clip_length`도 1로 그대로다. 특징이 1초 평균이다.

## 특징 (Ogawa 산출물, 읽기 전용 마운트)
| | 위치 | recipe | 컨테이너 경로 |
|---|---|---|---|
| 오디오 | `A_Ogawa_v1(0923)/runs/clotho_extract_20260925/features/audio` | `m2d2025-p80x2-timeframes-meanpool1s-v1` (M2D-CLAP 2025 p80x2 프레임 특징 1초 평균, CLAP 투영 없음, 768) | `/feat/m2d/clotho_audio` |
| 텍스트 | `A_Ogawa_v1(0923)/runs/gapcfg_20261007/text_m2d/clotho` | `m2dclap2025-p80x2-bert-last-hidden-state-v1` (BERT 텍스트 타워 token 출력, 투영 없음, 768) | `/feat/m2d/clotho_text` |

두 묶음 모두 체크포인트 `inputs/m2d/m2d_clap_vit_base-80x1001p80x2p16kpBpTI-2025/checkpoint-30.pth`(sha `ce8b9a…`)로 만들었다.

## 시작 전 확인 (2026-10-08)
- **ID:** cbbb5be 주석과 대조했다. 빠진 것은 없다.
  - train: 질의 32,694 / 녹음 27,472
  - val: 질의 4,918 / 녹음 4,107
- **키·차원:**
  - 오디오: `features` (60, 768) float32, 31,579개 전부
  - 텍스트: `last_hidden_state` 2차원 (토큰, 768) float32, 37,612개 전부. 토큰 수는 10~33(중앙값 15)이고, 32를 넘는 질의는 2개다.
- **시간 대응:**
  - 주석 duration은 전부 60이고, 정답 구간은 모두 [0, 60] 안에 있다.
  - 공식 `StartEndDataset`으로 읽으면 ctx_l × clip_length = 60 × 1 = 60이다. 정답 구간을 정규화했다가 다시 초로 바꾸면 원래 값과 같다(예: [41.1, 48.1]).
  - 오디오 입력은 768 + 시간 위치 특징 2 = 770차원이다. 배치 묶기(collate)도 정상이다.
- **Clotho padding:** 오디오가 전부 60프레임이라 audio padding이 없다.
- **resume:** `qd_resumable.py`는 `results/qd-baseline-fixed-20261008`의 것과 같다. 사전학습 경로는 수정 코드로 CPU에서 "중단 → 재개"한 결과가 한 번에 돌린 결과와 같았다.

## 실행과 재개
```sh
cd ~/dcase2026/results/qd-m2d-clotho-pretrain-20261008
./start_pretrain.sh 2024 2025                  # seed마다 컨테이너 하나. 시작과 재개 모두 이 명령
docker stop qd-m2d-clotho-pretrain-s<seed>    # 해당 seed 중지
```
결과는 `results/clotho_pretrain_m2d_s<seed>/`에 쌓인다. 로그는 seed 2023이 `logs/pretrain.log`, seed 2024·2025가 `logs/pretrain_s<seed>.log`다.

## 진행 방식
seed 2023의 첫 epoch들로 학습과 검증이 정상인지 확인한다. 확인할 항목은 loss 감소, val 지표, best checkpoint와 resume_state 저장이다. 이상이 있으면 seed 2024가 시작되기 전에 멈춘다.

## 진행 기록
- 06:22Z: seed를 차례로 도는 컨테이너 `qd-m2d-clotho-pretrain`로 시작했다. 첫 5 epoch에서 loss가 감소했고(3.35 → 1.74), val R1@0.7이 55.9 → 78.4로 올랐고, 체크포인트도 저장됐다. 정상이었다.
- 08:13Z: seed 2023을 완료했다(best epoch 135, Clotho val R1@0.7 84.42).
- 08:13Z: 사용자 요청으로 seed 2024·2025를 병렬로 돌리기로 했다.
  - 근거: 작업 하나의 SM 사용률이 약 38%였다.
  - 순차 컨테이너는 seed 2024 epoch 1 도중에 멈췄다(resume_state가 없어 처음부터 다시 시작).
  - seed마다 컨테이너(`qd-m2d-clotho-pretrain-s<seed>`)를 띄웠다. 이전 스크립트는 `logs/start_pretrain.sequential.sh.bak`에 있다.
  - 병렬 상태에서 seed당 50초/epoch이고, SM은 34% + 36%다. 두 seed 모두 약 11:00Z에 끝날 것으로 예상한다(순차였다면 약 11:55Z).
- 10:20Z: 사용자 요청으로 seed 2025를 잠시 멈췄다(docker stop, 150/200 epoch까지 resume_state 있음). seed 2023 CASTELLA 미세조정(`results/qd-m2d-castella-finetune-20261008`)을 먼저 돌린다.
  - `resume_2025_after_finetune_2023.sh`가 대기하다가 미세조정 컨테이너가 끝나면 `./start_pretrain.sh 2025`로 이어 간다. 로그는 `logs/resume_2025.log`, setsid로 분리 실행했다.
- 10:45Z: 사용자 요청으로 GPU0의 seed 2024(epoch 176 진행 중)와 2025(epoch 154 진행 중)를 docker stop했다. 직전 epoch까지 resume_state가 있고, `./start_pretrain.sh 2024 2025`로 이어서 한다.
- 10:5xZ: 수정 코드(cbbb5be)로 한 seed 2023 CASTELLA 미세조정 폴더 `results/qd-m2d-castella-finetune-20261008`은 사용자 지시로 지웠다. 기존 코드(481d60b)로 다시 하는 미세조정은 `results/qd-m2d-castella-finetune-481d60b-20261008`이다. 이 폴더의 사전학습 체크포인트를 그대로 쓴다.
