"""
Forward Test数据下载脚本

下载2026年1-2月的:
1. 1分钟K线数据 (data.binance.vision) → futures_data_1m_forward/
2. 资金费率数据 (Binance API) → funding_rates_forward/

月度归档优先，404时回退到逐日归档。
"""

import os
import io
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import threading
import calendar

# ============== 配置 ==============
CONFIG = {
    'kline_output_dir': './futures_data_1m_forward',
    'funding_output_dir': './funding_rates_forward',
    'start_year': 2026,
    'start_month': 1,
    'end_year': 2026,
    'end_month': 2,
    'interval': '1m',
    'max_workers': 3,
    'max_retries': 3,
    'timeout': 120,
}

# 复用 data_fetch.py 的 QUALIFIED_POOL
QUALIFIED_POOL = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'ASTERUSDT', 'TRUMPUSDT', 'BNBUSDT', 'SUIUSDT',
    'WIFUSDT', 'ADAUSDT', 'XPLUSDT', '1000SHIBUSDT', 'FARTCOINUSDT', 'WLFIUSDT', 'ENAUSDT', 'HYPEUSDT', 'COAIUSDT', 'AVAXUSDT',
    'LINKUSDT', 'MYXUSDT', 'NEIROUSDT', 'WLDUSDT', 'LTCUSDT', 'ORDIUSDT', 'PNUTUSDT', 'DOTUSDT', 'GMTUSDT', 'AVNTUSDT',
    'PUMPUSDT', 'GIGGLEUSDT', 'ETCUSDT', 'ARBUSDT', 'APTUSDT', '1000BONKUSDT', 'FILUSDT', 'PENGUUSDT', 'BCHUSDT', 'GALAUSDT',
    'APEUSDT', 'NEARUSDT', 'OPUSDT', 'SANDUSDT', 'TIAUSDT', 'MOODENGUSDT', 'AXSUSDT', 'NOTUSDT', '0GUSDT', 'ONDOUSDT',
    'VIRTUALUSDT', 'BOMEUSDT', 'KAITOUSDT', 'AAVEUSDT', 'LIGHTUSDT', 'UNIUSDT', 'MANAUSDT', '1000SATSUSDT', 'DYDXUSDT', 'ATOMUSDT',
    'SEIUSDT', 'SOMIUSDT', 'TAOUSDT', 'CRVUSDT', 'CFXUSDT', 'TRXUSDT', 'HUSDT', 'INJUSDT', 'PEOPLEUSDT', 'FETUSDT',
    'ETHFIUSDT', 'POPCATUSDT', 'ACTUSDT', 'TRBUSDT', 'XLMUSDT', 'CHZUSDT', 'SPKUSDT', '1000FLOKIUSDT', 'TONUSDT', 'FFUSDT',
    'MASKUSDT', '4USDT', 'ZECUSDT', 'MOVEUSDT', 'STBLUSDT', 'JUPUSDT', 'LINEAUSDT', 'GOATUSDT', 'PIPPINUSDT', 'LDOUSDT',
    'MUSDT', 'VINEUSDT', 'IPUSDT', 'OMUSDT', 'BLESSUSDT', 'EIGENUSDT', 'TURBOUSDT', 'ICPUSDT', 'STXUSDT', 'EVAAUSDT',
    'SUSHIUSDT', 'LAYERUSDT', 'ZORAUSDT', 'USUALUSDT', 'WCTUSDT', 'RUNEUSDT', 'HBARUSDT', 'BIOUSDT', 'DOGSUSDT', 'KITEUSDT',
    'MONUSDT', 'PAXGUSDT', 'JTOUSDT', 'ZROUSDT', '1MBABYDOGEUSDT', 'FUNUSDT', 'BERAUSDT', '1000LUNCUSDT', 'RIVERUSDT', 'ZBTUSDT',
    'AIXBTUSDT', 'MEMEUSDT', 'SAHARAUSDT', 'RESOLVUSDT', 'SOONUSDT', 'ENSUSDT', 'KGENUSDT', 'ALGOUSDT', 'IOUSDT', 'BIGTIMEUSDT',
    'JELLYJELLYUSDT', 'THETAUSDT', 'OPENUSDT', 'RAREUSDT', 'BASUSDT', 'HEMIUSDT', 'PROVEUSDT', 'GASUSDT', 'ARCUSDT', 'ALICEUSDT',
    'TRADOORUSDT', 'NXPCUSDT', 'HYPERUSDT', 'SAPIENUSDT', 'SAGAUSDT', 'TSTUSDT', 'VETUSDT', 'ATUSDT', 'LRCUSDT', 'BARDUSDT',
    'TRUTHUSDT', 'MEWUSDT', 'SPXUSDT', 'XTZUSDT', 'GRTUSDT', '1000RATSUSDT', 'MUBARAKUSDT', 'XPINUSDT', 'PENDLEUSDT', 'TAUSDT',
    'PYTHUSDT', 'LABUSDT', 'BANANAS31USDT', 'WUSDT', 'APRUSDT', 'JASMYUSDT', 'STRKUSDT', '2ZUSDT', 'RVVUSDT', 'BLURUSDT',
    'ZILUSDT', 'SUSDT', 'YFIUSDT', 'XNYUSDT', 'ZKCUSDT', 'MIRAUSDT', 'CUSDT', 'RECALLUSDT', 'ALCHUSDT', 'RENDERUSDT',
    'EDENUSDT', 'CLOUSDT', 'SWARMSUSDT', 'STORJUSDT', 'AUCTIONUSDT', 'PTBUSDT', 'TREEUSDT', 'ARKMUSDT', 'BBUSDT', 'YBUSDT',
    'THEUSDT', '1INCHUSDT', 'PARTIUSDT', 'COMPUSDT', 'IDUSDT', 'VANAUSDT', 'MANTAUSDT', 'MELANIAUSDT', 'ONEUSDT', 'SYRUPUSDT',
    'ENJUSDT', 'ALTUSDT', 'UBUSDT', 'NEOUSDT', 'XAIUSDT', 'SNXUSDT', 'MITOUSDT', 'EGLDUSDT', 'AEVOUSDT', 'LPTUSDT',
    'LUNA2USDT', 'COWUSDT', 'XANUSDT', 'MAGICUSDT', 'CYBERUSDT', 'YGGUSDT', 'LAUSDT', 'CAKEUSDT', 'INITUSDT', 'SOPHUSDT',
    'ZKUSDT', 'ANIMEUSDT', 'ERAUSDT', 'MTLUSDT', 'MERLUSDT', 'MEUSDT', 'USELESSUSDT', 'RSRUSDT', 'TNSRUSDT', 'CHRUSDT',
    'KNCUSDT', 'METUSDT', 'BABYUSDT', 'NEWTUSDT', 'HANAUSDT', 'ZEREBROUSDT', 'ARUSDT', 'IMXUSDT', 'ARKUSDT', 'KAVAUSDT',
    'CHILLGUYUSDT', 'QUSDT', 'POLUSDT', 'GRIFFAINUSDT', 'PIXELUSDT', 'OGNUSDT', 'HMSTRUSDT', 'POLYXUSDT', 'ALPINEUSDT', 'ZENUSDT',
    'XMRUSDT', 'CATIUSDT', 'API3USDT', 'GUNUSDT', 'BELUSDT', 'ENSOUSDT', 'HUMAUSDT', 'PROMPTUSDT', 'C98USDT', 'MINAUSDT',
    'REDUSDT', 'HOLOUSDT', 'CELOUSDT', 'TUTUSDT', 'CKBUSDT', 'RLCUSDT', 'LISTAUSDT', 'KSMUSDT', 'FORMUSDT', 'COTIUSDT',
    'COOKIEUSDT', 'UMAUSDT', 'TLMUSDT', 'BANUSDT', 'ANKRUSDT', 'HAEDALUSDT', 'INUSDT', 'AIUSDT', 'CELRUSDT', 'SUNUSDT',
    'SQDUSDT', 'QTUMUSDT', 'GRASSUSDT', 'BANDUSDT', 'REZUSDT', 'ZRXUSDT', 'DASHUSDT', 'ARPAUSDT', 'ROSEUSDT', '1000CATUSDT',
    'HIGHUSDT', 'ORCAUSDT', 'AGLDUSDT', 'AIOTUSDT', 'LITUSDT', 'FLOWUSDT', 'ZETAUSDT', 'DYMUSDT', 'HOTUSDT', 'TAKEUSDT',
    'IOTXUSDT', 'IOTAUSDT', 'SFPUSDT', 'GLMUSDT', 'KERNELUSDT', 'MORPHOUSDT', 'RVNUSDT', 'XVGUSDT', 'SKLUSDT', 'RAYSOLUSDT',
    'ORDERUSDT', 'ONTUSDT', 'DENTUSDT', 'MOCAUSDT', 'TURTLEUSDT', 'ACEUSDT', 'HIVEUSDT', 'SIGNUSDT', 'LQTYUSDT', 'BLUAIUSDT',
    'EDUUSDT', 'BMTUSDT', 'USTCUSDT', 'DAMUSDT', 'DOLOUSDT', 'FUSDT', 'COMMONUSDT', 'DRIFTUSDT', 'GTCUSDT', 'TOWNSUSDT',
    'ACHUSDT', 'CCUSDT', 'BUSDT', 'NMRUSDT', 'CROSSUSDT', 'BANANAUSDT', 'SSVUSDT', 'IOSTUSDT', 'STGUSDT', 'NAORISUSDT',
    'FHEUSDT', 'HIPPOUSDT', 'LYNUSDT', 'AERGOUSDT', 'VVVUSDT', 'CETUSUSDT', 'CTSIUSDT', 'AKEUSDT', 'HOMEUSDT', 'HOOKUSDT',
    'DIAUSDT', 'SUPERUSDT', 'KASUSDT', 'SHELLUSDT', 'ICNTUSDT', 'VFYUSDT', 'AVAAIUSDT', 'SOLVUSDT', 'ZKJUSDT', 'NILUSDT',
    'BATUSDT', 'PUMPBTCUSDT', 'OGUSDT', 'BEAMXUSDT', 'YALAUSDT', 'SPELLUSDT', 'ICXUSDT', 'A2ZUSDT', 'POWRUSDT', 'BTRUSDT',
    'NKNUSDT', 'AEROUSDT', 'ONGUSDT', 'RDNTUSDT', 'WALUSDT', 'BSVUSDT', 'WOOUSDT', 'BRETTUSDT', '1000000MOGUSDT', 'PORTALUSDT',
    'ATAUSDT', 'BIDUSDT', '1000CHEEMSUSDT', 'NFPUSDT', 'BROCCOLIF3BUSDT', 'EPICUSDT', 'PHBUSDT', 'MAVUSDT', 'ASRUSDT', 'SCRUSDT',
    'CGPTUSDT', 'TRUUSDT', 'STOUSDT', '1000XECUSDT', 'ESPORTSUSDT', 'DUSKUSDT', 'FIDAUSDT', 'TUSDT', 'ARIAUSDT', 'GPSUSDT',
    'AXLUSDT', 'VICUSDT', 'BROCCOLI714USDT', 'PLUMEUSDT', 'DMCUSDT', 'DOODUSDT', 'BANKUSDT', 'ASTRUSDT', '1000000BOBUSDT', 'DEXEUSDT',
    'BDXNUSDT', 'TOSHIUSDT', 'AGTUSDT', 'CUDISUSDT', 'LSKUSDT', 'KAIAUSDT', 'VTHOUSDT', 'CTKUSDT', 'SXTUSDT', 'MAVIAUSDT',
    'DEGENUSDT', 'SKYAIUSDT', 'DEEPUSDT', '42USDT', 'ATHUSDT', 'NTRNUSDT', 'EULUSDT', 'AIOUSDT', 'VANRYUSDT', 'KOMAUSDT',
    'BNTUSDT', 'JOEUSDT', 'VELVETUSDT', 'PLAYUSDT', 'MOVRUSDT', 'CVCUSDT', 'ONUSDT', 'MLNUSDT', 'GMXUSDT', 'SONICUSDT',
    'IDOLUSDT', 'HEIUSDT', 'STEEMUSDT', 'FLOCKUSDT', 'TACUSDT', 'AINUSDT', 'SAFEUSDT', 'SYNUSDT', 'PHAUSDT', 'QNTUSDT',
    'B2USDT', 'SCRTUSDT', 'HFTUSDT', 'NOMUSDT', 'SKYUSDT', 'AUSDT', 'DEGOUSDT', 'XVSUSDT', 'PUNDIXUSDT',
]

KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]

MONTHLY_KLINE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
DAILY_KLINE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
FUNDING_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


# ============== 进度追踪 ==============
class ProgressTracker:
    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.label = label

    def update(self, status: str, info: str = ""):
        with self.lock:
            self.completed += 1
            if status == 'success':
                self.success += 1
            elif status == 'failed':
                self.failed += 1
            elif status == 'skipped':
                self.skipped += 1
            self._print_progress(info, status)

    def _print_progress(self, info: str, status: str):
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0

        bar_len = 30
        filled = int(bar_len * self.completed / self.total)
        bar = '=' * filled + '-' * (bar_len - filled)

        status_icon = {'success': '+', 'failed': 'x', 'skipped': 'o'}.get(status, '?')

        sys.stdout.write(f'\r[{self.label}] [{bar}] {self.completed}/{self.total} '
                        f'| +{self.success} x{self.failed} o{self.skipped} '
                        f'| ETA: {eta:.0f}s | {status_icon} {info:<25}')
        sys.stdout.flush()


# ============== K线数据下载 ==============

def _download_zip(url: str) -> Optional[pd.DataFrame]:
    """下载并解析zip文件中的CSV"""
    for attempt in range(CONFIG['max_retries']):
        try:
            resp = requests.get(url, timeout=CONFIG['timeout'])
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    temp_df = pd.read_csv(f, header=None)
                    first_val = str(temp_df.iloc[0, 0])
                    if not first_val.replace('.', '').replace('-', '').isdigit():
                        temp_df = temp_df.iloc[1:].reset_index(drop=True)
                    temp_df.columns = KLINE_COLUMNS[:len(temp_df.columns)]
                    return temp_df
        except Exception:
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(1 * (attempt + 1))
    return None


def download_month_klines(symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """先尝试月度归档，404时逐日下载"""
    interval = CONFIG['interval']

    # 1) 尝试月度归档
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{MONTHLY_KLINE_URL}/{symbol}/{interval}/{filename}"
    df = _download_zip(url)
    if df is not None and len(df) > 0:
        return df

    # 2) 回退到逐日归档
    days_in_month = calendar.monthrange(year, month)[1]
    today = datetime.utcnow().date()
    daily_dfs = []

    for day in range(1, days_in_month + 1):
        date = datetime(year, month, day).date()
        if date >= today:  # 今天及未来的日期跳过
            break
        day_filename = f"{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
        day_url = f"{DAILY_KLINE_URL}/{symbol}/{interval}/{day_filename}"
        day_df = _download_zip(day_url)
        if day_df is not None and len(day_df) > 0:
            daily_dfs.append(day_df)
        time.sleep(0.01)

    if daily_dfs:
        return pd.concat(daily_dfs, ignore_index=True)
    return None


def download_symbol_klines(symbol: str, output_dir: Path,
                           months: List[Tuple[int, int]],
                           progress: ProgressTracker) -> Tuple[str, str]:
    """下载单个标的的所有月份K线数据"""
    output_path = output_dir / f"{symbol}.parquet"

    if output_path.exists():
        progress.update('skipped', symbol)
        return symbol, 'skipped'

    all_dfs = []
    for year, month in months:
        df = download_month_klines(symbol, year, month)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
        time.sleep(0.02)

    if not all_dfs:
        progress.update('failed', f"{symbol} (no data)")
        return symbol, 'no_data'

    combined = pd.concat(all_dfs, ignore_index=True)

    # 时间戳处理
    combined['open_time'] = pd.to_numeric(combined['open_time'], errors='coerce')
    if combined['open_time'].iloc[0] > 1e15:
        combined['open_time'] = pd.to_datetime(combined['open_time'], unit='us')
    else:
        combined['open_time'] = pd.to_datetime(combined['open_time'], unit='ms')

    # 删除不需要的列
    drop_cols = ['close_time', 'ignore']
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    combined['symbol'] = symbol

    # 转换类型
    float_cols = ['open', 'high', 'low', 'close', 'volume',
                  'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
    for col in float_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce').astype('float32')

    if 'trades' in combined.columns:
        combined['trades'] = pd.to_numeric(combined['trades'], errors='coerce').astype('int32')

    # 去重排序
    combined = combined.drop_duplicates(subset=['open_time']).sort_values('open_time')

    # 列顺序
    output_cols = ['open_time', 'symbol', 'open', 'high', 'low', 'close',
                   'volume', 'quote_volume', 'trades',
                   'taker_buy_volume', 'taker_buy_quote_volume']
    combined = combined[[c for c in output_cols if c in combined.columns]]

    combined.to_parquet(output_path, compression='snappy', index=False)

    rows = len(combined)
    size_mb = output_path.stat().st_size / 1024 / 1024
    progress.update('success', f"{symbol} ({rows:,}, {size_mb:.0f}MB)")
    return symbol, 'success'


# ============== 资金费率下载 ==============

def download_symbol_funding(symbol: str, output_dir: Path,
                            start_ms: int, end_ms: int,
                            progress: ProgressTracker) -> Tuple[str, str]:
    """通过Binance API下载单个标的的资金费率"""
    output_path = output_dir / f"{symbol}.parquet"

    if output_path.exists():
        progress.update('skipped', symbol)
        return symbol, 'skipped'

    all_records = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000,
        }
        for attempt in range(CONFIG['max_retries']):
            try:
                resp = requests.get(FUNDING_API_URL, params=params, timeout=60)
                if resp.status_code == 400:
                    # 该symbol可能不存在或参数错误
                    break
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt < CONFIG['max_retries'] - 1:
                    time.sleep(1 * (attempt + 1))
                data = []

        if not data:
            break

        all_records.extend(data)

        # 下一页: 最后一条的时间+1ms
        last_time = data[-1].get('fundingTime', 0)
        if last_time <= current_start:
            break
        current_start = last_time + 1

        if len(data) < 1000:
            break

        time.sleep(0.1)  # API限速

    if not all_records:
        progress.update('failed', f"{symbol} (no data)")
        return symbol, 'no_data'

    df = pd.DataFrame(all_records)
    # 标准化列名以匹配现有 funding_rates/ 格式
    df['calc_time'] = pd.to_datetime(pd.to_numeric(df['fundingTime']), unit='ms')
    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df['mark_price'] = pd.to_numeric(df.get('markPrice', pd.Series(dtype=float)), errors='coerce')
    df['symbol'] = symbol

    df = df[['calc_time', 'funding_rate', 'mark_price', 'symbol']].sort_values('calc_time')
    df = df.drop_duplicates(subset=['calc_time'])

    df.to_parquet(output_path, compression='snappy', index=False)

    progress.update('success', f"{symbol} ({len(df)} rates)")
    return symbol, 'success'


# ============== 主函数 ==============

def download_klines():
    """下载1分钟K线数据"""
    output_dir = Path(CONFIG['kline_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    months = []
    for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
        s_m = CONFIG['start_month'] if year == CONFIG['start_year'] else 1
        e_m = CONFIG['end_month'] if year == CONFIG['end_year'] else 12
        for month in range(s_m, e_m + 1):
            months.append((year, month))

    print("=" * 70)
    print("Forward Test: 1分钟K线数据下载")
    print("=" * 70)
    print(f"票池: {len(QUALIFIED_POOL)} symbols")
    print(f"时间: {months[0][0]}-{months[0][1]:02d} ~ {months[-1][0]}-{months[-1][1]:02d}")
    print(f"输出: {output_dir.absolute()}")
    print(f"策略: 月度归档优先, 日度归档回退")

    existing = sum(1 for s in QUALIFIED_POOL if (output_dir / f"{s}.parquet").exists())
    print(f"已完成: {existing}, 待下载: {len(QUALIFIED_POOL) - existing}")
    print("-" * 70)

    progress = ProgressTracker(len(QUALIFIED_POOL), "Kline")
    results = {}

    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(download_symbol_klines, symbol, output_dir, months, progress): symbol
            for symbol in QUALIFIED_POOL
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, status = future.result()
                results[sym] = status
            except Exception as e:
                results[symbol] = f'error: {str(e)[:50]}'

    print(f"\n\nK线下载完成: +{progress.success} x{progress.failed} o{progress.skipped}")
    return results


def download_funding_rates():
    """下载资金费率数据"""
    output_dir = Path(CONFIG['funding_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(CONFIG['start_year'], CONFIG['start_month'], 1)
    end_dt = datetime(CONFIG['end_year'], CONFIG['end_month'],
                      calendar.monthrange(CONFIG['end_year'], CONFIG['end_month'])[1],
                      23, 59, 59)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print("\n" + "=" * 70)
    print("Forward Test: 资金费率数据下载")
    print("=" * 70)
    print(f"票池: {len(QUALIFIED_POOL)} symbols")
    print(f"时间: {start_dt.date()} ~ {end_dt.date()}")
    print(f"输出: {output_dir.absolute()}")

    existing = sum(1 for s in QUALIFIED_POOL if (output_dir / f"{s}.parquet").exists())
    print(f"已完成: {existing}, 待下载: {len(QUALIFIED_POOL) - existing}")
    print("-" * 70)

    progress = ProgressTracker(len(QUALIFIED_POOL), "Funding")
    results = {}

    # 资金费率API有限速，使用较少线程
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(download_symbol_funding, symbol, output_dir,
                          start_ms, end_ms, progress): symbol
            for symbol in QUALIFIED_POOL
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, status = future.result()
                results[sym] = status
            except Exception as e:
                results[symbol] = f'error: {str(e)[:50]}'

    print(f"\n\n资金费率下载完成: +{progress.success} x{progress.failed} o{progress.skipped}")
    return results


def main():
    print("Forward Test Data Download")
    print(f"Time: {datetime.now().isoformat()}\n")

    # 1. 下载K线数据
    kline_results = download_klines()

    # 2. 下载资金费率
    funding_results = download_funding_rates()

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    kline_dir = Path(CONFIG['kline_output_dir'])
    kline_files = list(kline_dir.glob("*.parquet"))
    kline_size = sum(f.stat().st_size for f in kline_files) / 1024 / 1024 / 1024
    print(f"K线: {len(kline_files)} files, {kline_size:.2f} GB")

    funding_dir = Path(CONFIG['funding_output_dir'])
    funding_files = list(funding_dir.glob("*.parquet"))
    funding_size = sum(f.stat().st_size for f in funding_files) / 1024 / 1024
    print(f"资金费率: {len(funding_files)} files, {funding_size:.1f} MB")

    # 日志
    log_path = kline_dir / "download_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Range: {CONFIG['start_year']}-{CONFIG['start_month']:02d} ~ "
                f"{CONFIG['end_year']}-{CONFIG['end_month']:02d}\n\n")
        f.write("=== Kline Results ===\n")
        for symbol, status in sorted(kline_results.items()):
            f.write(f"{symbol}: {status}\n")
        f.write("\n=== Funding Rate Results ===\n")
        for symbol, status in sorted(funding_results.items()):
            f.write(f"{symbol}: {status}\n")

    kline_failed = sum(1 for v in kline_results.values() if v not in ('success', 'skipped'))
    funding_failed = sum(1 for v in funding_results.values() if v not in ('success', 'skipped'))
    if kline_failed > 0 or funding_failed > 0:
        print(f"\nWarning: {kline_failed} kline + {funding_failed} funding failures. Re-run to retry.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Re-run to continue (supports resume).")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
