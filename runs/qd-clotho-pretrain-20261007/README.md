# QD-DETR baseline — Clotho-Moment 사전학습 → CASTELLA 미세조정 (2026-10-07 예약)

사용자 지시(2026-10-07): 공식 baseline QD-DETR을 MS-CLAP 특징으로 Clotho-Moment에서 사전학습하는 작업을 GPU0 예약 대기로 건다. 목적은 공식 README의 "Clotho-Moment pre-training & CASTELLA fine-tuning" 행(test R1@0.7 13.85 ± 1.47, 5회)을 우리 환경에서 재현하고, 기존 CASTELLA만 학습한 10 seed 결과(`legacy/task6-qd-detr/results/castella_10seeds_summary.tsv`, test R1@0.7 9.70 ± 1.45)와 비교하는 것이다.

## 무엇을 돌리나

- 코드: `legacy/task6-qd-detr`(rev `481d60b`)의 `src/`를 **수정 없이** import한다. `qd_resumable.py`는 `train.main()+train()`과 같은 순서로 학습하고, epoch마다 전체 상태를 저장하는 부분만 더했다.
- 이 문서는 당시 실행 기록이다. 이후 현재 브랜치에 공식 attention mask 및 CASTELLA 주석 수정이 병합되었다. 이미 생성된 `DONE.json`, 예측, 지표는 수정 전 결과이며, 미완료 단계를 이어 실행할 때는 해당 실행의 코드·데이터 버전(`481d60b`)을 유지해야 한다.
- seed 2023–2027을 차례로 돈다. seed마다 다음 순서다.
  1. `P_s<seed>`: Clotho-Moment 사전학습. `config_pretraining.yml` 그대로(200 epoch, lr 1e-4, batch 32, 매 epoch Clotho val 4,918개 평가, best val R1@0.7 checkpoint)
  2. `F_s<seed>`: CASTELLA 미세조정. `config.yml` 그대로이며, P의 best checkpoint 가중치로 시작한다(공식 `--resume`과 같은 방식). 10 seed CASTELLA-only 실행의 `configs/castella_seed<seed>.yml`과 results_dir만 다르다.
  3. F best checkpoint로 val(352)·test(1,347) 추론 → `results/F_s<seed>/eval_{val,test}/`
- 바뀐 설정은 seed, results_dir, P의 특징 경로뿐이다(`make_configs.sh`).
- 특징: Clotho-Moment는 공식 Zenodo 17129257 MS-CLAP(오디오 51,240 / 텍스트 44,261, 768차원, `A_Kim_v1(0923)/server/downloads/...`, 읽기 전용 마운트). CASTELLA는 baseline 폴더의 기존 특징이다.
- 주석: baseline 폴더의 release jsonl(공식 test 1,347 질의). 공통 평가셋(1,333)이나 annotations_v2가 아니다.
- **test는 seed마다 한 번만 채점한다.** `eval_test/DONE.json`이 있으면 다시 돌리지 않는다.
- 컨테이너: `dcase26-task6:py271-cu128`, GPU0, `--cpus 4`, `--network none`.

## 예약 규칙 (`reserve.py`)

- GPU0의 CV5 fold 컨테이너(`t0004-cv5*`)가 1개 이하이고, 다른 프로세스가 1 GiB 미만이고, 여유 메모리가 4 GiB 이상인 상태가 10초 간격 3회 연속이면 시작한다.
- 시작하면 CV5 큐는 이 작업을 외부 작업으로 보고 GPU0에 새 fold를 올리지 않는다. 이미 돌던 fold는 계속 돈다. 이 체인이 끝나면 CV5가 GPU0 2슬롯을 다시 쓴다.
- 다섯 seed는 컨테이너 하나 안에서 한 프로세스로 이어서 돈다. 단계 사이에 GPU가 비는 틈을 줄이기 위해서다.

## 예상 시간

- 측정(10-07 03:41Z, GPU0 단독): 사전학습 35.5초/epoch → 200 epoch 약 2시간
- 미세조정은 기존 실행 기준 약 25분이다. seed당 약 2시간 25분, 5 seed 약 12시간이다(단독 기준). CV5 fold와 같이 돌면 더 느리다.

## resume 검증 (CPU, 작은 부분집합, 10-07)

scratchpad `qd_resume_test/`에서 Clotho 48/16, CASTELLA 24/8/8 질의, 3 epoch, num_workers 2로 시험했다.

- "P epoch 2에서 중단 → 재개, F epoch 2에서 중단 → 재개"한 결과가 한 번에 돌린 결과와 같았다. model·optimizer·best checkpoint 텐서와 train/val log(타임스탬프 제외)가 모두 일치했다.
- 공식 `src/train.py`로 같은 설정을 돌린 결과와도 best checkpoint 텐서와 epoch별 log가 일치했다. 공식 `src/evaluate.py`와는 지표와 submission 파일이 일치했다.
- 완료된 eval 단계는 다시 실행하면 건너뛴다(재채점 없음).

## 명령

```sh
cd ~/dcase2026/results/qd-clotho-pretrain-20261007
# 예약 시작 / 재부팅 후 재개 (같은 명령)
setsid --fork python3 reserve.py >> logs/reserve.log 2>&1 < /dev/null
# 새 시작 막기: touch HOLD  (돌던 컨테이너는 계속 돈다. 멈추려면 docker stop qd-clotho-pt-chain → 다시 시작하면 이어서 함)
tail logs/reserve.log logs/chain.log
cat results/*/DONE.json results/F_s*/eval_*/DONE.json
```

## 상태

- 2026-10-07: 예약 시작(아래 reserve.log). 시작 시점 GPU0는 CV5 fold 2개(cp-s2024-f0, i3-s2024-f0)가 차지하고 있다.
