import requests, os, json
from datetime import datetime
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def fetch_exchange_rate_by_yfinance():
    os.makedirs('exchange_rate', exist_ok=True)
    filename = f'exchange_rate/{datetime.now().strftime("%Y%m%d")}.csv'
    if not os.path.exists(filename):
        import yfinance as yf
        df = yf.download(['USDKRW=X']).reset_index()
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
    
    return df.iloc[-1, -2]

    
# https://app.exchangerate-api.com/
def fetch_exchange_rate_by_exchange_rate_com():
    # 오늘 날짜를 기준으로 파일 이름 생성
    today_date_str = datetime.utcnow().strftime('%Y%m%d.json')

    # 오늘 날짜의 환율 파일이 있는지 확인
    if not os.path.exists(today_date_str):
        # 파일이 없으면 API 요청
        EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')
        url = f'https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/USD'
        
        response = requests.get(url)
        data = response.json()
        
        # 파일 저장
        with open(os.path.join('exchange_rate', today_date_str), 'w') as f:
            json.dump(data, f)
    else:
        # 파일이 있으면 로드
        with open(os.path.join('exchange_rate', today_date_str), 'r') as f:
            data = json.load(f)
    
    return data

def convert_usd_to_krw(usd):
    exchange_info = fetch_exchange_rate_by_yfinance()
    print("┌" + "─" * 30 + "┐")
    content = f"{usd * exchange_info:,.2f}원 사용"
    print(f"│{content.center(28)}│")
    print("└" + "─" * 30 + "┘")
    return usd * exchange_info

