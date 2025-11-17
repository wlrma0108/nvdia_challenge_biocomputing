# 🚀 빠른 시작 가이드 (.txt 파일 버전)

## ✅ 1단계: 파일 확인

```bash
# 데이터 파일이 있는지 확인
python scripts/00_verify_geo_files.py
```

**예상 출력:**
```
================================================================================
🔍 GEO DATA FILES VERIFICATION
================================================================================

📁 Scanning: data/geo_datasets

Found files:
  .txt files: 6
  .txt.gz files: 0

Total: 6 files

📄 GSE164416_series_matrix.txt
   Size: 15.23 MB
   Lines read: 500
   Title: Affymetrix profiling of IMIDIA biobank...
   Samples: ~133
   Has table: ✓ Yes

📄 GSE76894_series_matrix.txt
   Size: 8.45 MB
   Samples: ~103
   Has table: ✓ Yes

...

✅ NEXT STEPS
🎉 All 6 files look good!
Ready to run:
  python scripts/15_process_geo_robust.py
```

---

## 🔬 2단계: 데이터 처리 (메인 파이프라인)

```bash
python scripts/15_process_geo_robust.py
```

**처리 과정 (약 2-5분):**
```
🔬 PRODUCTION-GRADE GEO DATA PROCESSING PIPELINE v2.0
================================================================================

📥 STEP 1: LOADING GEO DATASETS (6개 파일 파싱)
📄 Parsing: GSE76894_series_matrix.txt
  ✓ Expression: 29,530 probes × 103 samples
  🏷️ Extracting labels...
    Label distribution:
      Diabetes: 19 (18.4%)
      Control: 84 (81.6%)
✓ Saved: figures/01_parsing_summary.png

🎯 STEP 2: SELECTING BEST DATASET
🏆 Selected: GSE76894 (가장 많은 valid samples)

🔍 STEP 3: QUALITY CONTROL
1️⃣ Removing high-missing probes (>20%)
2️⃣ Imputing remaining values
3️⃣ Removing low-variance probes
✓ Saved: figures/03_quality_control.png

📊 STEP 4: NORMALIZATION
  Log2 transformation + Z-score normalization
✓ Saved: figures/04_normalization.png

✂️ STEP 5: TRAIN/VAL/TEST SPLIT
  Train: 72 samples
  Val:   16 samples
  Test:  15 samples
✓ Saved: figures/05_data_split.png

💾 STEP 6: SAVING PROCESSED DATA
✓ Saved to data/real_geo_processed/

✅ PROCESSING COMPLETE!
```

---

## 📊 3단계: 결과 확인

### 생성된 파일들:

```bash
data/real_geo_processed/
├── X_train.csv          # 훈련 데이터 (72 × ~25,000)
├── X_val.csv            # 검증 데이터 (16 × ~25,000)
├── X_test.csv           # 테스트 데이터 (15 × ~25,000)
├── y_train.csv          # 훈련 레이블
├── y_val.csv            # 검증 레이블
├── y_test.csv           # 테스트 레이블
├── gene_names.csv       # 프로브/유전자 이름
├── PROCESSING_SUMMARY.txt  # 요약 리포트
│
├── GSE76894_raw_expression.csv   # 백업
├── GSE76894_raw_metadata.csv     # 백업
│
└── figures/             # 📊 시각화 결과 (5개 PNG)
    ├── 01_parsing_summary.png
    ├── 02_data_overview.png
    ├── 03_quality_control.png
    ├── 04_normalization.png
    └── 05_data_split.png
```

### 시각화 확인:

```bash
# Windows
start data/real_geo_processed/figures/

# Mac
open data/real_geo_processed/figures/

# Linux
xdg-open data/real_geo_processed/figures/
```

---

## 🎯 4단계: ML 파이프라인 실행

이제 기존 파이프라인을 실제 데이터로 실행하세요!

### 옵션 A: 경로만 수정해서 실행

기존 스크립트들의 데이터 경로를 수정:

```python
# scripts/05_feature_selection.py 수정
# 기존:
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# 변경:
PROCESSED_DIR = BASE_DIR / 'data' / 'real_geo_processed'
```

그 다음 순서대로 실행:
```bash
python scripts/05_feature_selection.py    # 특징 선택
python scripts/06_ml_models.py            # 모델 훈련
python scripts/10_clinical_grade_optimization.py  # 임상 최적화
```

### 옵션 B: 새 스크립트 작성 (권장)

실제 데이터용 새 스크립트를 만들어서 실행하는 것을 추천드립니다.

---

## ⚠️ 문제 해결

### 문제 1: "No valid samples found"

**원인:** 레이블 추출 실패 (모두 "Unknown")

**해결:**
```bash
# 메타데이터 확인
head -20 data/real_geo_processed/GSE76894_raw_metadata.csv

# 'characteristics_ch1' 또는 'title' 컬럼에 diabetes/control 키워드 확인
# 없다면 수동으로 레이블 파일 생성 필요
```

### 문제 2: "0 probes" 또는 "No expression data"

**원인:** Expression table 파싱 실패

**해결:**
```bash
# 원본 파일 확인
grep -n "series_matrix_table_begin" data/geo_datasets/GSE76894_series_matrix.txt

# 테이블이 파일 끝부분에 있을 수 있음
# 스크립트가 처음 500줄만 읽는다면 놓칠 수 있음
```

스크립트는 **전체 파일**을 읽으므로 문제없지만, 파일이 손상되었을 수 있습니다.

### 문제 3: "Memory Error"

**해결:** 큰 데이터셋의 경우
```python
# scripts/15_process_geo_robust.py 수정
# Line ~370: 샘플 수 제한
max_samples = 200  # 메모리 부족시
```

---

## 📈 예상 결과

### GSE76894 기준:

```
원본 데이터:
- 103개 샘플 (19 T2D, 84 ND)
- 29,530개 프로브 (Affymetrix)

QC 후:
- 103개 샘플 유지
- ~25,000개 프로브 (저품질 제거)

Train/Val/Test 분할:
- Train: 72 samples (13 diabetes, 59 control)
- Val:   16 samples (3 diabetes, 13 control)
- Test:  15 samples (3 diabetes, 12 control)

특징 선택 후:
- 50-100개 핵심 바이오마커

예상 성능:
- ROC-AUC: 0.70-0.80 (실제 환자 데이터)
- 민감도: 60-80% (시뮬레이션보다 낮지만 현실적)
- 특이도: 70-85%
```

---

## 💡 추가 팁

### Probe ID → Gene Symbol 변환

더 나은 해석을 위해 유전자 이름으로 변환:

```bash
# 1. Platform annotation 다운로드
cd data/geo_datasets
wget ftp://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570/GPL570.annot.gz
gunzip GPL570.annot.gz

# 2. 스크립트에 매핑 코드 추가
# (추후 업데이트 예정)
```

### 여러 데이터셋 통합

```python
# 여러 데이터셋 결합 (고급)
# scripts/20_combine_datasets.py (작성 필요)
# - GSE164416 + GSE76894 통합
# - ComBat으로 배치 효과 보정
# - 샘플 수 증가 → 성능 향상
```

---

## ✅ 체크리스트

- [ ] `python scripts/00_verify_geo_files.py` 실행 → 모든 파일 OK
- [ ] `python scripts/15_process_geo_robust.py` 실행 완료
- [ ] `data/real_geo_processed/` 폴더 확인
- [ ] 5개 시각화 PNG 파일 확인
- [ ] `PROCESSING_SUMMARY.txt` 읽어보기
- [ ] ML 파이프라인 실행 (scripts/05, 06, 10)
- [ ] 결과 분석 및 시뮬레이션과 비교

---

**마지막 업데이트:** 2025-11-17
**작성자:** Claude
**버전:** 2.0 (.txt 파일 최적화)

**문의사항이나 오류 발생 시 PROCESSING_SUMMARY.txt와 함께 보고해주세요!**
