#  这边由于下载速度不知道为什么很慢，用了多线程
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
import threading
from queue import Queue
from tqdm import tqdm

# --- 配置 ---
OUTPUT_DIR = os.path.join('data', 'raw')
ZH_FILENAME = os.path.join(OUTPUT_DIR, "chinese_data.txt")
START_URL = "https://news.sina.com.cn/"
TARGET_SIZE_MB = 5.0
MAX_WORKERS = 20

# --- 全局变量 ---
urls_to_visit = Queue()
urls_to_visit.put(START_URL)
visited_urls = set([START_URL])
file_lock = threading.Lock()
pbar = None
stop_event = threading.Event()

# --- 爬虫逻辑 ---
LINK_PATTERN = r'https?://(news|finance|tech|sports)\.sina\.com\.cn/.*?/\d{4}-\d{2}-\d{2}/doc-.*\.shtml'
ARTICLE_BODY_SELECTORS = ['.article-content-left', '#article', '.main-content', '.article-body']
HEADERS = {'User-Agent': 'Mozilla/5.0 ...'}

def get_text_and_links(url):
    try:
        # 添加 proxies 参数以忽略系统代理, 电脑自带代理会卡住所以这么设置
        response = requests.get(url, headers=HEADERS, timeout=10, proxies={'http': None, 'https': None})
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article_text = None
        for selector in ARTICLE_BODY_SELECTORS:
            article_body = soup.select_one(selector)
            if article_body:
                for s in article_body.select('script, style'): s.decompose()
                paragraphs = article_body.find_all('p')
                text_content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if len(text_content) > 100:
                    article_text = text_content
                    break
        
        new_links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if re.match(LINK_PATTERN, urljoin(url, a['href']))]
        return article_text, list(set(new_links))
    except requests.exceptions.RequestException:
        return None, []

def worker():
    while not stop_event.is_set():
        try:
            current_url = urls_to_visit.get(timeout=1)
        except Exception:
            continue

        article_text, new_links = get_text_and_links(current_url)
        
        if article_text:
            with file_lock:
                with open(ZH_FILENAME, 'a', encoding='utf-8') as f: f.write(article_text + "\n\n")
                current_size_mb = os.path.getsize(ZH_FILENAME) / (1024 * 1024)
                pbar.n = current_size_mb
                pbar.refresh()
                if current_size_mb >= TARGET_SIZE_MB: stop_event.set()

        with file_lock:
            for link in new_links:
                if link not in visited_urls:
                    visited_urls.add(link)
                    urls_to_visit.put(link)
        urls_to_visit.task_done()

def crawl_chinese_data_fast():
    global pbar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(ZH_FILENAME): os.remove(ZH_FILENAME)

    print("开始从新浪新闻并发爬取数据...")
    threads = [threading.Thread(target=worker) for _ in range(MAX_WORKERS)]
    for t in threads: t.start()

    with tqdm(total=TARGET_SIZE_MB, unit='MB', desc="爬取进度") as progress_bar:
        pbar = progress_bar
        stop_event.wait()

    print("\n已达到目标大小，正在优雅地停止所有线程...")
    for t in threads: t.join(timeout=2)
    print(f"中文数据爬取完成。最终文件大小: {os.path.getsize(ZH_FILENAME) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    crawl_chinese_data_fast()