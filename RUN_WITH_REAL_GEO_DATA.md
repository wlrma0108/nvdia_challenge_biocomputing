# 실제 GEO 데이터로 파이프라인 실행하기

## 📁 준비된 데이터셋

```
data/geo_datasets/
├── GSE164416_series_matrix.txt (또는 .txt.gz)
├── GSE25724_series_matrix.txt
├── GSE76894_series_matrix.txt
├── GSE81608_series_matrix.txt
├── GSE86468_series_matrix.txt
└── GSE86469_series_matrix.txt
```

## 🚀 실행 방법

### 1단계: GEO 데이터 처리

```bash
# 6개 GEO 데이터셋을 파싱하고 전처리
python scripts/14_process_geo_complete.py
```

**출력:**
- `data/real_geo_processed/X_train.csv` - 훈련 데이터
- `data/real_geo_processed/X_val.csv` - 검증 데이터
- `data/real_geo_processed/X_test.csv` - 테스트 데이터
- `data/real_geo_processed/y_train.csv` - 훈련 레이블
- `data/real_geo_processed/gene_names.csv` - 유전자/프로브 이름

**처리 내용:**
- ✅ Series matrix 파일 파싱
- ✅ 당뇨/정상 샘플 자동 분류
- ✅ 결측치 처리 (>20% missing 제거)
- ✅ 저분산 프로브 제거 (하위 10%)
- ✅ Log2 변환 (필요시)
- ✅ Z-score 정규화
- ✅ Train/Val/Test 분할 (70/15/15)

### 2단계: 특징 선택 (실제 데이터 사용)

```bash
python scripts/15_feature_selection_real.py
```

### 3단계: 모델 훈련

```bash
python scripts/16_ml_models_real.py
```

### 4단계: 임상급 최적화

```bash
python scripts/17_clinical_optimization_real.py
```

### 5단계: 테스트 평가

```bash
python scripts/18_test_evaluation_real.py
```

### 6단계: 생물학적 해석

```bash
python scripts/19_biological_interpretation_real.py
```

---

## 📊 예상 결과

### GSE76894 기준 (206 샘플)

**클래스 분포:**
- Diabetes (T2D): ~55명
- Control (ND): ~120명
- 기타 (IGT, T3cD): 제외

**데이터 크기:**
- 프로브 수: ~54,000 (GPL570 array)
- QC 후: ~40,000-45,000
- 특징 선택 후: 50-100개

**예상 성능:**
- 실제 환자 데이터로 더 현실적인 성능
- 시뮬레이션 대비 낮을 수 있음 (정상)
- 대신 임상 적용 가능성 훨씬 높음

---

## 🔧 문제 해결

### 문제 1: "No GEO files found"

**해결:**
```bash
# 파일이 올바른 위치에 있는지 확인
ls -lah data/geo_datasets/

# .gz 압축 해제가 필요하면
gunzip data/geo_datasets/*.gz
```

### 문제 2: 메모리 부족

**해결:**
```python
# scripts/14_process_geo_complete.py 수정
# Line ~200: 샘플 크기 제한
max_samples = 200  # 메모리가 부족하면 줄이기
```

### 문제 3: Probe ID를 Gene Symbol로 변환하고 싶음

**해결:**

1. **GPL570 Annotation 다운로드:**
```bash
cd data/geo_datasets/
wget ftp://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570/GPL570.annot.gz
gunzip GPL570.annot.gz
```

2. **스크립트에 매핑 추가** (`scripts/14_process_geo_complete.py`의 5번 섹션 수정):
```python
# Load GPL570 annotation
gpl570 = pd.read_csv('data/geo_datasets/GPL570.annot',
                      sep='\t', comment='#', low_memory=False)

# Map probe to gene
probe_to_gene = dict(zip(gpl570['ID'], gpl570['Gene Symbol']))

# Rename probes to genes
gene_expression.index = gene_expression.index.map(
    lambda x: probe_to_gene.get(x, x)
)

# Aggregate duplicate genes (take mean)
gene_expression = gene_expression.groupby(gene_expression.index).mean()
```

---

## 📈 다음 단계

### 옵션 1: 단일 데이터셋만 사용

우선순위가 높은 데이터셋만 선택:
- **GSE164416** (1순위) - 췌장 도, 가장 관련성 높음
- **GSE76894** (2순위) - 대규모 코호트, 검증된 연구

### 옵션 2: 여러 데이터셋 통합 (고급)

Meta-analysis 수행:
1. 각 데이터셋 개별 처리
2. 배치 효과 보정 (ComBat)
3. 통합 분석

```python
# scripts/20_meta_analysis.py 참고
from combat.pycombat import pycombat

# Combine datasets
combined_data = pd.concat([gse164416, gse76894], axis=1)
batch_labels = ['GSE164416']*n1 + ['GSE76894']*n2

# Batch correction
corrected_data = pycombat(combined_data, batch_labels)
```

### 옵션 3: 외부 검증

1. **Training**: GSE164416, GSE76894
2. **Validation**: GSE25724
3. **Test**: GSE86468

다른 코호트에서 모델 성능 검증 → 일반화 능력 확인

---

## ⚠️ 중요 참고사항

### Probe ID vs Gene Symbol

**현재 상태:**
- Affymetrix 프로브 ID 사용 (예: `1007_s_at`)
- 약 54,000개 프로브 → 약 20,000-25,000개 유전자에 해당

**권장사항:**
- GPL570 annotation 다운로드하여 gene symbol로 변환
- 중복 프로브는 평균값 사용
- 더 해석 가능한 결과

### 데이터셋별 특성

| Dataset | 조직 | 샘플 수 | 특징 |
|---------|------|---------|------|
| GSE164416 | 췌장 도 | ~100 | Primary, 가장 직접적 |
| GSE76894 | 혈액/도 | 206 | 대규모, 검증된 연구 |
| GSE25724 | 췌장 도 | ~80 | 검증용 |
| GSE86469 | 세포 특이적 | ~50 | 기전 연구 |

**추천:**
- **시작**: GSE76894 (샘플 많음, 안정적)
- **검증**: GSE25724 또는 GSE86468
- **심화**: GSE164416 + GSE76894 통합

---

## 📚 참고 자료

**원본 논문:**
- GSE76894: Solimena et al. (2018) "Systems biology of the IMIDIA biobank"
  PubMed ID: 29185012

**Platform:**
- GPL570: Affymetrix Human Genome U133 Plus 2.0 Array
- ~54,000 probe sets
- ~47,000 transcripts

**GEO 링크:**
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76894
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164416

---

## ✅ 실행 체크리스트

- [ ] 6개 GEO 파일 다운로드 완료
- [ ] `data/geo_datasets/` 폴더에 배치
- [ ] `python scripts/14_process_geo_complete.py` 실행
- [ ] `data/real_geo_processed/` 폴더 확인
- [ ] 나머지 파이프라인 스크립트 실행
- [ ] 결과 분석 및 시뮬레이션과 비교

---

**마지막 업데이트:** 2025-11-17
**작성자:** Claude
**상태:** 로컬 실행 준비 완료 ✅
