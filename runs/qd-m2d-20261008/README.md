# QD-DETR, M2D-CLAP 특징: Clotho-Moment 사전학습 → CASTELLA 미세조정 (2026-10-08)

MS-CLAP 대신 M2D-CLAP 특징(오디오 `m2d2025-p80x2-timeframes-meanpool1s-v1`, 텍스트 `m2dclap2025-p80x2-bert-last-hidden-state-v1`, 768차원)으로 공식 QD-DETR을 학습한 기록이다. 텍스트 파일만 넣었고, 체크포인트·`resume_state.pt`·학습 로그·특징 manifest·Clotho val 예측은 뺐다.

| 단계 | 폴더 | 코드 | seed | 결과 |
|---|---|---|---|---|
| Clotho-Moment 사전학습 | `clotho_pretrain/` | cbbb5be | 2023 | best epoch 135, Clotho val R1@0.7 84.42 |
| CASTELLA 미세조정 | `castella_finetune_481d60b/` | 481d60b | 2023 | best epoch 140, val R1@0.7 33.24 / test R1@0.5 43.96, R1@0.7 25.58, mAP 20.95 |

- 미세조정은 MS-CLAP 실행(`../qd-clotho-pretrain-20261007/`)과 같은 기존 코드 481d60b와 그 주석으로 돌렸다. 공식 수정(bf0e88f attention mask, 35fcbaf CASTELLA 타임스탬프)은 들어 있지 않다.
- 사전학습은 수정 코드 cbbb5be로 돌렸지만 그 체크포인트를 그대로 썼다. Clotho-Moment 오디오는 전부 60프레임이라 오디오 패딩이 없고, 이때 mask 수정 전후의 attention mask가 같다(전부 False).
- 평가 질의는 M2D 특징이 있는 질의만이다: train 2,158 / val 343 / test 1,333(공통 평가셋과 같은 test 질의). 뺀 47개는 `castella_finetune_481d60b/data/excluded_queries.json`에 있다. 따라서 공식 test 1,347 질의로 채점한 MS-CLAP 표와 바로 비교할 수 없다.
- test는 한 번만 채점했다.
- seed 2024·2025 사전학습은 각각 epoch 176·154에서 멈춘 상태라 아직 여기에 없다.
