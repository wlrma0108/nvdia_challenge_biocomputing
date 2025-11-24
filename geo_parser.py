import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GEOSeriesMatrixParser:

    
    def __init__(self, filepath: str):
        """초기화"""
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.geo_id = self.filename.split('_')[0]
        
        self.metadata: Dict[str, List[str]] = {}
        self.expression_data: Optional[pd.DataFrame] = None
        self.sample_info: Optional[pd.DataFrame] = None
        self.diabetes_labels: Optional[List[str]] = None
        
        self.n_genes = 0
        self.n_samples = 0
        self.is_log_scaled = None
    
    def parse(self, verbose: bool = True) -> bool:
  
        if verbose:
            print(f"\n{'='*80}")
            print(f"📂 Parsing: {self.filename}")
            print(f"{'='*80}")
        
        try:
            # 1. 파일 읽기
            metadata_lines, data_lines = self._read_file()
            
            self._parse_metadata(metadata_lines)
            if verbose:
                self._print_metadata_summary()
            
            self._parse_expression_data(data_lines)
            if verbose:
                self._print_expression_summary()
            
            # 4. 샘플 정보 추출
            self._extract_sample_info()
            if verbose:
                self._print_sample_summary()
            
            # 5. 당뇨 라벨 감지
            self._detect_diabetes_labels()
            if verbose:
                self._print_label_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error parsing {self.filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_file(self) -> Tuple[List[str], List[List[str]]]:
        """파일을 읽어서 메타데이터와 데이터 섹션으로 분리"""
        metadata_lines = []
        data_lines = []
        data_started = False
        
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                # 메타데이터 라인 (! 시작)
                if line.startswith('!'):
                    metadata_lines.append(line)
                
                # 데이터 헤더 (ID_REF 시작)
                elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                    data_started = True
                    headers = [h.strip('"').strip() for h in line.split('\t')]
                    data_lines.append(headers)
                
                # 데이터 행
                elif data_started:
                    data_lines.append(line.split('\t'))
        
        return metadata_lines, data_lines
    
    def _parse_metadata(self, lines: List[str]) -> None:
        """메타데이터 파싱"""
        for line in lines:
            parts = line.split('\t')
            key = parts[0].replace('!', '').strip()
            values = [v.strip('"').strip() for v in parts[1:] if v.strip()]
            
            if values:
                self.metadata[key] = values
    
    def _parse_expression_data(self, lines: List[List[str]]) -> None:
        """발현 데이터 매트릭스 파싱"""
        if not lines or len(lines) < 2:
            print("⚠️  No expression data found")
            return
        
        headers = lines[0]
        data_rows = []
        
        # 헤더 길이와 맞는 행만 선택
        for row in lines[1:]:
            if len(row) == len(headers):
                data_rows.append(row)
        
        # DataFrame 생성
        df = pd.DataFrame(data_rows, columns=headers)
        
        # 인덱스 설정
        if 'ID_REF' in df.columns:
            df.set_index('ID_REF', inplace=True)
        
        # 숫자로 변환
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.expression_data = df
        self.n_genes = df.shape[0]
        self.n_samples = df.shape[1]
        
        # Log scale 감지
        self._detect_log_scale()
    
    def _detect_log_scale(self) -> None:
        """데이터가 log-scaled인지 감지"""
        if self.expression_data is None:
            return
        
        max_val = self.expression_data.values.max()
        min_val = self.expression_data.values.min()
        
        if min_val >= 0 and max_val < 50:
            self.is_log_scaled = True
        elif max_val > 100:
            self.is_log_scaled = False
        else:
            self.is_log_scaled = None
    
    def _extract_sample_info(self) -> None:
        """샘플 메타데이터를 DataFrame으로 추출"""
        sample_data = {}
        
        # 샘플 ID 가져오기
        if 'Sample_geo_accession' not in self.metadata:
            print("⚠️  No sample IDs found")
            return
        
        sample_ids = self.metadata['Sample_geo_accession']
        sample_data['sample_id'] = sample_ids
        
        # 모든 Sample_ 메타데이터 추출
        for key, values in self.metadata.items():
            if key.startswith('Sample_') and key != 'Sample_geo_accession':
                if len(values) == len(sample_ids):
                    clean_key = key.replace('Sample_', '').lower()
                    sample_data[clean_key] = values
        
        self.sample_info = pd.DataFrame(sample_data)
    
    def _detect_diabetes_labels(self) -> None:
        """샘플 메타데이터에서 당뇨 라벨 자동 감지"""
        if self.sample_info is None or len(self.sample_info) == 0:
            return
        
        labels = []
        
        # 각 그룹의 키워드
        t2dm_keywords = ['t2dm', 't2d', 'type 2', 'type2', 'diabetic', 'diabetes', 'dep']
        igt_keywords = ['igt', 'impaired glucose tolerance', 'impaired glucose', 
                       'prediabetes', 'pre-diabetic', 'pre diabetic']
        t3c_keywords = ['t3cd', 't3c', 'type 3c', 'type3c']
        control_keywords = ['nd', 'non-diabetic', 'non diabetic', 'normal', 
                           'control', 'healthy']
        
        # 각 샘플에 대해 모든 컬럼 검색
        for idx, row in self.sample_info.iterrows():
            # 모든 값을 문자열로 변환하고 검색
            row_text = ' '.join(str(v).lower() for v in row.values)
            
            # 구체성 순서로 체크
            if any(kw in row_text for kw in igt_keywords):
                labels.append('IGT')
            elif any(kw in row_text for kw in t3c_keywords):
                labels.append('T3cD')
            elif any(kw in row_text for kw in t2dm_keywords):
                labels.append('T2DM')
            elif any(kw in row_text for kw in control_keywords):
                labels.append('Control')
            else:
                labels.append('Unknown')
        
        self.diabetes_labels = labels
        self.sample_info['diabetes_label'] = labels
    
    def _print_metadata_summary(self) -> None:
        """메타데이터 요약 출력"""
        print(f"\n📋 METADATA:")
        
        if 'Series_title' in self.metadata:
            title = self.metadata['Series_title'][0]
            print(f"   Title: {title[:70]}...")
        
        if 'Series_geo_accession' in self.metadata:
            print(f"   GEO ID: {self.metadata['Series_geo_accession'][0]}")
        
        if 'Series_platform_id' in self.metadata:
            print(f"   Platform: {self.metadata['Series_platform_id'][0]}")
    
    def _print_expression_summary(self) -> None:
        """발현 데이터 요약 출력"""
        if self.expression_data is None:
            return
        
        print(f"\n📊 EXPRESSION MATRIX:")
        print(f"   Shape: {self.n_genes:,} genes × {self.n_samples} samples")
        print(f"   Mean: {self.expression_data.values.mean():.3f}")
        print(f"   Range: [{self.expression_data.values.min():.3f}, "
              f"{self.expression_data.values.max():.3f}]")
        
        # 결측치
        missing = self.expression_data.isnull().sum().sum()
        total = self.expression_data.size
        print(f"   Missing: {missing:,} ({100*missing/total:.2f}%)")
        
        # 0 값
        zeros = (self.expression_data == 0).sum().sum()
        print(f"   Zeros: {zeros:,} ({100*zeros/total:.2f}%)")
        
        # Log scale
        if self.is_log_scaled is True:
            print(f"   Scale: ✓ LOG-SCALED")
        elif self.is_log_scaled is False:
            print(f"   Scale: LINEAR")
        else:
            print(f"   Scale: UNCERTAIN")
    
    def _print_sample_summary(self) -> None:
        """샘플 정보 요약 출력"""
        if self.sample_info is None:
            return
        
        print(f"\n👥 SAMPLES: {len(self.sample_info)}")
        print(f"   Metadata fields: {len(self.sample_info.columns)}")
    
    def _print_label_summary(self) -> None:
        """라벨 감지 결과 출력"""
        if self.diabetes_labels is None:
            return
        
        print(f"\n🎯 DIABETES LABELS:")
        label_counts = pd.Series(self.diabetes_labels).value_counts()
        
        for label, count in label_counts.items():
            if label != 'Unknown':
                print(f"   ✓ {label}: {count} samples")
        
        if 'Unknown' in label_counts:
            print(f"   ⚠️  Unknown: {label_counts['Unknown']} samples")
    
    def get_expression_matrix(self) -> pd.DataFrame:
        """발현 매트릭스 반환"""
        return self.expression_data
    
    def get_sample_metadata(self) -> pd.DataFrame:
        """샘플 메타데이터 반환"""
        return self.sample_info
    
    def get_labels(self) -> List[str]:
        """당뇨 라벨 반환"""
        return self.diabetes_labels
    
    def save_to_csv(self, output_dir: str = '.') -> None:
        """
        파싱된 데이터를 CSV 파일로 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 발현 데이터 저장 (처음 1000개 유전자만)
        if self.expression_data is not None:
            expr_file = os.path.join(output_dir, f'{self.geo_id}_expression.csv')
            self.expression_data.head(1000).to_csv(expr_file)
            print(f"💾 Saved: {expr_file}")
        
        # 샘플 메타데이터 저장
        if self.sample_info is not None:
            meta_file = os.path.join(output_dir, f'{self.geo_id}_metadata.csv')
            self.sample_info.to_csv(meta_file, index=False)
            print(f"💾 Saved: {meta_file}")


def parse_all_geo_files(input_dir: str, output_dir: str = './parsed_data') -> Dict[str, GEOSeriesMatrixParser]:
    """
    디렉토리 내 모든 GEO series matrix 파일 파싱
    
    Args:
        input_dir: GEO 파일이 있는 디렉토리
        output_dir: 파싱된 데이터 저장 디렉토리
        
    Returns:
        GEO ID를 키로 하는 파서 딕셔너리
    """
    print("\n" + "="*80)
    print("🧬 GEO SERIES MATRIX BATCH PARSER")
    print("="*80)
    
    # Series matrix 파일 찾기
    files = [f for f in os.listdir(input_dir) if f.endswith('_series_matrix.txt')]
    
    if not files:
        print(f"❌ No series matrix files found in {input_dir}")
        return {}
    
    print(f"\n📁 Found {len(files)} files:")
    for f in files:
        print(f"   • {f}")
    
    # 각 파일 파싱
    parsers = {}
    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        parser = GEOSeriesMatrixParser(filepath)
        
        if parser.parse(verbose=True):
            parser.save_to_csv(output_dir)
            parsers[parser.geo_id] = parser
    
    # 요약 출력
    print("\n" + "="*80)
    print("📊 PARSING SUMMARY")
    print("="*80)
    
    summary_data = []
    for geo_id, parser in parsers.items():
        if parser.expression_data is not None:
            summary_data.append({
                'GEO_ID': geo_id,
                'Genes': parser.n_genes,
                'Samples': parser.n_samples,
                'Control': parser.diabetes_labels.count('Control') if parser.diabetes_labels else 0,
                'IGT': parser.diabetes_labels.count('IGT') if parser.diabetes_labels else 0,
                'T2DM': parser.diabetes_labels.count('T2DM') if parser.diabetes_labels else 0,
                'T3cD': parser.diabetes_labels.count('T3cD') if parser.diabetes_labels else 0,
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(f"\n{summary_df.to_string(index=False)}")
        
        # 요약 저장
        summary_file = os.path.join(output_dir, 'parsing_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ ALL PARSING COMPLETE!")
    print("="*80)
    
    return parsers


# 사용 예시
if __name__ == '__main__':
    # 모든 파일 파싱
    parsers = parse_all_geo_files(
        input_dir='data',
        output_dir='outputdata'
    )
    
    # 개별 데이터셋 접근
    if 'GSE164416' in parsers:
        gse164416 = parsers['GSE164416']
        expr = gse164416.get_expression_matrix()
        metadata = gse164416.get_sample_metadata()
        labels = gse164416.get_labels()
        
        print(f"\n🎯 GSE164416 loaded:")
        print(f"  Expression shape: {expr.shape}")
        print(f"  Label distribution:")
        print(f"    {pd.Series(labels).value_counts().to_dict()}")