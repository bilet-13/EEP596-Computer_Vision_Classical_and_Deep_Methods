import pandas as pd
import requests
import os
import time
import re
import random  # 新增 random 用來做隨機延遲

INPUT_FILE = 'billboard_albums.xlsx'   
OUTPUT_DIR = 'dataset'

COL_ARTIST = 'artist_name'
COL_ALBUM = 'album_name'
COL_YEAR = 'list_year'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br"
}

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name))

def get_decade_folder(year):
    try:
        year_int = int(str(year)[:4])
        decade = (year_int // 10) * 10
        return f"{decade}"
    except:
        return None

def fetch_album_cover(artist, album, save_path):
    term = f"{album}" 
    
    url = "https://itunes.apple.com/search"
    params = {
        "term": term,
        "media": "music",
        "entity": "album",
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        
        if response.status_code == 403:
            print(f"\n⚠️ status code 403...")
            time.sleep(10)
            return False

        if response.status_code == 200:
            data = response.json()
            if data['resultCount'] > 0:
                img_url = data['results'][0]['artworkUrl100'].replace("100x100bb", "600x600bb")
                
                img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
                with open(save_path, 'wb') as f:
                    f.write(img_data)
                return True
            else:
                print(f"  -> API return album not found: {album}")
                
    except Exception as e:
        print(f"Error fetching {album}: {e}")
    
    return False

def main():
    print("正在讀取 Excel...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"找不到 {INPUT_FILE}，請確認檔名是否正確。")
        return

    print(f"共發現 {len(df)} 筆資料。開始下載...")

    success_count = 0
    fail_count = 0


    for index, row in df.iterrows():
        if index % 50 == 0 and index != 0:
            print("rest 5 seconds...")
            time.sleep(5)

        artist = row[COL_ARTIST]
        album = row[COL_ALBUM]
        year = row[COL_YEAR]

        if pd.isna(artist) or pd.isna(album) or pd.isna(year):
            continue

        decade = get_decade_folder(year)
        if not decade: 
            continue 
            
        folder_path = os.path.join(OUTPUT_DIR, decade)
        os.makedirs(folder_path, exist_ok=True)

        filename = sanitize_filename(f"{artist}_{album}_{year}.jpg")
        save_path = os.path.join(folder_path, filename)

        if os.path.exists(save_path):
            print(f"[跳過] 已存在: {year} - {album}")
            continue

        print(f"[{index+1}/{len(df)}] 下載中: {album}", end="\r")
        
        if fetch_album_cover(artist, album, save_path):
            success_count += 1
            print(f"{success_count} ✅ 成功 ({year}): {album}") 
        else:
            fail_count += 1
            print(f"{fail_count} ❌ 失敗: {album}")
        
        sleep_time = random.uniform(1.0, 3.0)
        time.sleep(sleep_time)

    print(f"\n\n下載完成！")
    print(f"成功: {success_count} 張")
    print(f"失敗/未找到: {fail_count} 張")

if __name__ == "__main__":
    main()
