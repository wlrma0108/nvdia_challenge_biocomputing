# 실제 GEO 데이터셋 사용 가이드

## 문제 상황
현재 환경에서 NCBI GEO 서버 접근이 제한되어 자동 다운로드가 불가능합니다 (403 Forbidden 오류).

## 추천 데이터셋

### Core Datasets (필수)
1. **GSE164416** - Pancreatic islet samples
   - 목적: Primary training data
   - 조직: Pancreatic islets
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164416

2. **GSE76894** - Blood samples
   - 목적: Blood-based biomarker discovery
   - 조직: Whole blood
   - 대규모 코호트 (~200 samples)
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76894

3. **GSE86469** - Cell-type resolution
   - 목적: Mechanism understanding
   - 단일세포 resolution
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86469

4. **GSE81608** - Cell-type resolution
   - 목적: Mechanism understanding
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81608

### Validation Datasets (검증용)
5. **GSE25724** - External validation
   - 목적: Independent validation
   - 조직: Pancreatic islets
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25724

6. **GSE86468** - Additional validation
   - 목적: Additional validation
   - GEO 링크: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86468

## 방법 1: GEO 웹사이트에서 수동 다운로드

### 단계별 가이드:

1. **GEO 웹사이트 접속**
   ```
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164416
   ```

2. **Series Matrix File 다운로드**
   - 페이지 하단의 "Download family" 섹션 찾기
   - "Series Matrix File(s)" 클릭하여 다운로드
   - 파일명: `GSE164416_series_matrix.txt.gz`

3. **Supplementary Files 다운로드 (선택사항)**
   - Raw data나 processed data가 필요한 경우
   - "Supplementary file" 섹션에서 다운로드

4. **다운로드한 파일을 프로젝트에 복사**
   ```bash
   # 다운로드한 파일을 프로젝트 디렉토리로 복사
   mkdir -p data/geo_datasets
   cp ~/Downloads/GSE164416_series_matrix.txt.gz data/geo_datasets/
   ```

5. **데이터 로드 및 처리**
   ```python
   import pandas as pd
   import gzip

   # Unzip and read
   with gzip.open('data/geo_datasets/GSE164416_series_matrix.txt.gz', 'rt') as f:
       # Parse the series matrix file
       lines = f.readlines()
       # Extract expression data and metadata
   ```

## 방법 2: GEOquery R 패키지 사용

GEOquery는 R에서 더 안정적으로 작동합니다:

```r
# R에서 실행
library(GEOquery)

# 데이터셋 다운로드
gse <- getGEO("GSE164416", GSEMatrix = TRUE, destdir = "data/geo_datasets")

# Expression data 추출
expr_data <- exprs(gse[[1]])
write.csv(expr_data, "data/geo_datasets/GSE164416_expression.csv")

# Sample metadata 추출
pheno_data <- pData(gse[[1]])
write.csv(pheno_data, "data/geo_datasets/GSE164416_metadata.csv")
```

## 방법 3: 로컬 환경에서 Python으로 다운로드

제한이 없는 로컬 환경에서:

```python
import GEOparse
import pandas as pd

# 각 데이터셋 다운로드
datasets = ['GSE164416', 'GSE76894', 'GSE86469', 'GSE81608', 'GSE25724', 'GSE86468']

for geo_id in datasets:
    print(f"Downloading {geo_id}...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir='data/geo_datasets')

    # Save expression matrix
    # (코드는 scripts/00_download_real_geo_datasets.py 참조)
```

## 방법 4: GEO2R 온라인 도구 사용

1. GEO 페이지에서 "Analyze with GEO2R" 클릭
2. 온라인에서 샘플 그룹 정의 및 분석
3. 결과를 다운로드하여 사용

## 다운로드 후 분석 파이프라인 실행

데이터를 다운로드한 후, 다음 스크립트들을 순서대로 실행:

```bash
# 1. 데이터 전처리
python scripts/02_preprocessing.py

# 2. 탐색적 데이터 분석
python scripts/03_exploratory_analysis.py

# 3. 차등 발현 분석
python scripts/04_differential_expression.py

# 4. 특징 선택
python scripts/05_feature_selection.py

# 5. ML 모델 학습
python scripts/06_ml_models.py

# 6. 모델 평가
python scripts/07_model_evaluation.py
```

## 데이터 전처리 시 주의사항

### 1. 샘플 메타데이터 확인
```python
# Control vs Diabetes 그룹 식별
metadata = pd.read_csv('data/geo_datasets/GSE164416_metadata.csv')
print(metadata['characteristics_ch1'].value_counts())
```

### 2. 배치 효과 확인
```python
# 여러 데이터셋을 합칠 때 배치 효과 고려
# ComBat 또는 다른 batch correction 방법 적용
```

### 3. 플랫폼 호환성
```python
# 서로 다른 플랫폼의 데이터를 합칠 때
# Gene symbol로 통일하거나 공통 유전자만 사용
```

## 데이터셋별 특징

| 데이터셋 | 조직 | 샘플 수 | 플랫폼 | 용도 |
|---------|------|---------|--------|------|
| GSE164416 | Islet | Medium | RNA-seq | Primary training |
| GSE76894 | Blood | ~200 | Microarray | Biomarker |
| GSE86469 | Multiple | Medium | scRNA-seq | Mechanism |
| GSE81608 | Multiple | Medium | scRNA-seq | Mechanism |
| GSE25724 | Islet | ~40 | Microarray | Validation |
| GSE86468 | Multiple | Medium | RNA-seq | Validation |

## 현재 시뮬레이션 데이터로 완성된 파이프라인

현재는 생물학적으로 현실적인 시뮬레이션 데이터로 전체 파이프라인이 구축되어 있습니다:

✅ 완료된 단계:
1. 데이터 생성 (20,000 genes, 150 samples, 39 known diabetes genes)
2. 전처리 및 정규화
3. 탐색적 데이터 분석 (PCA, t-SNE, heatmaps)
4. 차등 발현 분석
5. 특징 선택 (50 biomarkers identified, 6 known genes recovered)
6. ML 모델 학습 (7 models with hyperparameter tuning)
7. 모델 평가 (진행 중)

## 실제 데이터 적용 시 장점

실제 GEO 데이터를 사용하면:
1. ✅ 실제 생물학적 변이 반영
2. ✅ 발표 가능한 결과
3. ✅ 외부 검증 데이터셋으로 robustness 확인
4. ✅ 알려진 바이오마커와 비교 가능

## 문의사항

데이터 다운로드나 분석에 문제가 있으면:
1. GEO 웹사이트에서 데이터 availability 확인
2. 데이터셋 문서 읽기 (README, Series record)
3. 원저자 논문 참조

## 참고 자료

- GEO: https://www.ncbi.nlm.nih.gov/geo/
- GEOquery Bioconductor: https://bioconductor.org/packages/release/bioc/html/GEOquery.html
- GEOparse Python: https://geoparse.readthedocs.io/
