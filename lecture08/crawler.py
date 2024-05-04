import supabase
from dotenv import load_dotenv
import urllib.parse
import multiprocessing
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import sys
from queue import Queue
from multiprocessing import Manager
import platform
from glob import glob
import os

# 환경변수 설정
load_dotenv("../.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


# 드라이버 경로 설정
os_name = platform.system().lower()
architecture = platform.machine()

DRIVER_PATH = None
if os_name == 'darwin': # 맥 사용자
    DRIVER_PATH = glob('./driver/**/chromedriver', recursive=True)[0]
else: # 윈도우 사용자
    DRIVER_PATH = glob('./driver/**/chromedriver.exe', recursive=True)[0]



service = Service(executable_path=DRIVER_PATH)
chrome_options =  webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--blink-settings=imagesEnabled=false')

def fetch_data(index, links_slice, data_queue):
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # 이부분을 추가합니다.(이미지 크롤링을 위함입니다.) ################
        driver.set_window_size(800, 3000)
        driver.execute_script("document.body.style.zoom='10%'")
        ###########################################################
        for url in tqdm(links_slice, desc=f"#{index} 크롤러 수집률"):
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.select_one("title").getText() if soup.select_one("title") else ""
            description = soup.select_one('meta[name="description"]')['content'] if soup.select_one('meta[name="description"]') else ""
            canonical_url = soup.select_one('link[rel="canonical"]')['href'] if soup.select_one('link[rel="canonical"]') else ""
            address = soup.select_one('.css-1mwesgd').getText() if soup.select_one('.css-1mwesgd') else ""
            image = soup.select_one('section[aria-label="갤러리"] img')['src'] if soup.select_one('section[aria-label="갤러리"] img') else ""
            id = urllib.parse.urlparse(canonical_url).path.split('/')[-1]
            
            data_queue.put((id, title, description, canonical_url, address, image))
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        driver.quit()

def store_data(data_queue):
    sb = supabase.create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    id, title, description, canonical_url, address, image = data_queue.get()
    sb.table('scraps').insert([{'id': id, 'title': title, 'description': description, 'url': canonical_url, 'address': address, 'image': image}]).execute()
    
    
    seen_ids = set()
    try:
        total_items = data_queue.qsize()
        for _ in tqdm(range(total_items), desc="DB 저장 진행"):
            data = data_queue.get()
            if data[0] not in seen_ids:
                id, title, description, canonical_url, address, image = data
                sb.table('scraps').insert([{'id': id, 'title': title, 'description': description, 'url': canonical_url, 'address': address, 'image': image}]).execute()
                seen_ids.add(id)
    except Exception as e:
        print(f"Error occurred during DB insertion: {e}")

def main_multiprocess(links, num_processes):
    manager = Manager()
    data_queue = manager.Queue()
    chunk_size = len(links) // num_processes
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunks = [links[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
        pool.starmap(fetch_data, [(i, chunk, data_queue) for i, chunk in enumerate(chunks)])
    
    store_data(data_queue)

def get_links() :
    sb = supabase.create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    data = sb.table('links').select('url').execute().data
    links = [row['url'] for row in data]
    return links


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python crawler.py [프로세스 수]")
        sys.exit(1)
    num_processes = int(sys.argv[1])
    links = get_links()
    main_multiprocess(links, num_processes)
