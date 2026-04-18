"""
GitHub Actions 每日自動執行
不需要 token，直接寫入 weather.json
"""
import json, requests
from datetime import datetime

REGIONS = {
    'taipei':     (25.048, 121.531),
    'new_taipei': (25.012, 121.465),
    'taoyuan':    (24.993, 121.301),
    'hsinchu':    (24.807, 120.968),
    'miaoli':     (24.560, 120.820),
}

def fetch(lat, lon):
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&daily=temperature_2m_max,temperature_2m_min,"
           "precipitation_sum,precipitation_probability_max,weathercode"
           "&timezone=Asia/Taipei&forecast_days=7")
    d = requests.get(url, timeout=15).json()['daily']
    return [{'tmax': round(d['temperature_2m_max'][i]),
             'tmin': round(d['temperature_2m_min'][i]),
             'rain': round(d['precipitation_sum'][i] * 10) / 10,
             'rainProb': round(d['precipitation_probability_max'][i] or 0),
             'wcode': d['weathercode'][i]} for i in range(7)]

weather = {}
for region, (lat, lon) in REGIONS.items():
    print(f"抓取 {region}...", end=" ", flush=True)
    try:
        weather[region] = fetch(lat, lon)
        print(f"✓ {weather[region][0]['tmax']}°/{weather[region][0]['tmin']}°")
    except Exception as e:
        print(f"✗ {e}")

weather['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')

with open('weather.json', 'w', encoding='utf-8') as f:
    json.dump(weather, f, ensure_ascii=False, indent=2)

print(f"✓ weather.json 更新完成：{weather['updated']}")
