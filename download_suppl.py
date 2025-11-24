import os
import urllib.request
from pathlib import Path
import time

def download_file_with_retry(url, output_path, max_retries=3):
    
    for attempt in range(max_retries):
        try:
            print(f"      Attempt {attempt + 1}/{max_retries}...")
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(output_path, 'wb') as f:
                    downloaded = 0
                    block_size = 8192
                    
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        
                        downloaded += len(buffer)
                        f.write(buffer)
                        
                        # 진행상황 표시
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r      Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                
                print()  # 줄바꿈
                
                # 파일 크기 확인
                if os.path.getsize(output_path) > 0:
                    return True
                else:
                    print(f"      ⚠️ Downloaded file is empty")
                    
        except Exception as e:
            print(f"\n      ❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"      ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                return False
    
    return False

def download_geo_supplementary_v2(geo_id, output_dir='./suppl_data'):
    """개선된 GEO supplementary files 다운로드"""
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    geo_dir = Path(output_dir) / geo_id
    geo_dir.mkdir(exist_ok=True)
    
    # GEO FTP 경로 구성
    geo_num = geo_id.replace('GSE', '')
    geo_series = f"GSE{geo_num[:-3]}nnn"
    
    print(f"🔍 Checking {geo_id}...")
    
    try:
        from ftplib import FTP
        ftp = FTP('ftp.ncbi.nlm.nih.gov')
        ftp.login()
        
        ftp_path = f"/geo/series/{geo_series}/{geo_id}/suppl/"
        ftp.cwd(ftp_path)
        
        # 파일 목록 가져오기
        filenames = []
        ftp.retrlines('NLST', filenames.append)
        
        if not filenames:
            print(f"   ⚠️  No supplementary files found")
            ftp.quit()
            return False
        
        print(f"   📦 Found {len(filenames)} file(s)")
        
        success_count = 0
        
        for filename in filenames:
            local_path = geo_dir / filename
            
            # 이미 존재하고 크기가 0보다 크면 스킵
            if local_path.exists() and local_path.stat().st_size > 0:
                print(f"   ⏭️  Skipping {filename} (already exists)")
                success_count += 1
                continue
            
            print(f"   ⬇️  Downloading {filename}...")
            
            # HTTP URL로 다운로드 (더 안정적)
            http_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series}/{geo_id}/suppl/{filename}"
            
            if download_file_with_retry(http_url, local_path):
                file_size = local_path.stat().st_size / (1024*1024)  # MB
                print(f"   ✅ Saved: {filename} ({file_size:.2f} MB)")
                success_count += 1
            else:
                print(f"   ❌ Failed to download {filename}")
                if local_path.exists():
                    local_path.unlink()  # 실패한 파일 삭제
        
        ftp.quit()
        
        if success_count > 0:
            print(f"   🎉 Successfully downloaded {success_count}/{len(filenames)} file(s)")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == '__main__':
    geo_ids = ['GSE164416', 'GSE81608', 'GSE86468', 'GSE86469']
    
    print("="*80)
    print("📥 GEO Supplementary Files Downloader v2")
    print("="*80)
    print()
    
    results = {}
    
    for geo_id in geo_ids:
        success = download_geo_supplementary_v2(geo_id)
        results[geo_id] = success
        print()
    
    print("="*80)
    print("📊 Download Summary")
    print("="*80)
    
    for geo_id, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {geo_id}: {status}")
    
    print("="*80)