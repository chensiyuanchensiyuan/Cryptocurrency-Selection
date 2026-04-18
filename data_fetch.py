"""
Binance U本位永续合约 1分钟K线数据下载脚本 (扩展版)

- 459个币种（日均quote_volume前90%）
- 时间范围: 2022-01-01 ~ 2025-12-31
- 每个标的一个文件，支持断点续传
- 输出: parquet格式（压缩存储）

筛选标准：
- 日均quote_volume >= 10%分位数（约$11.75M，去掉流动性最差的10%）

目录结构:
    futures_data_1m/
    ├── BTCUSDT.parquet
    ├── ETHUSDT.parquet
    └── ...

使用方法：
    python data_fetch.py
"""

import os
import io
import sys
import time
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import threading

# ============== 配置 ==============
CONFIG = {
    'output_dir': './futures_data_1m',
    'start_year': 2022,
    'start_month': 1,
    'end_year': 2025,
    'end_month': 12,
    'interval': '1m',
    'max_workers': 3,
    'max_retries': 3,
    'timeout': 120,
}

# 459个币种（日均quote_volume前90%，按流动性排序）
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

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
MARK_PRICE_URL = "https://data.binance.vision/data/futures/um/monthly/markPriceKlines"

KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
]

MARK_PRICE_COLUMNS = [
    'open_time', 'mark_open', 'mark_high', 'mark_low', 'mark_close',
    'ignore1', 'close_time', 'ignore2', 'ignore3', 
    'ignore4', 'ignore5', 'ignore6'
]


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
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
        bar = '█' * filled + '░' * (bar_len - filled)
        
        status_icon = {'success': '✓', 'failed': '✗', 'skipped': '⊘'}.get(status, '?')
        
        sys.stdout.write(f'\r[{bar}] {self.completed}/{self.total} '
                        f'| ✓{self.success} ✗{self.failed} ⊘{self.skipped} '
                        f'| ETA: {eta:.0f}s | {status_icon} {info:<25}')
        sys.stdout.flush()


def generate_month_list() -> List[Tuple[int, int]]:
    """生成月份列表"""
    months = []
    now = datetime.now()
    
    for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
        s_m = CONFIG['start_month'] if year == CONFIG['start_year'] else 1
        e_m = CONFIG['end_month'] if year == CONFIG['end_year'] else 12
        
        for month in range(s_m, e_m + 1):
            if year > now.year or (year == now.year and month > now.month):
                continue
            months.append((year, month))
    
    return months


def download_single_month(symbol: str, year: int, month: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """下载单个月的数据"""
    interval = CONFIG['interval']
    kline_df = None
    mark_df = None
    
    # 下载K线数据
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    
    for attempt in range(CONFIG['max_retries']):
        try:
            resp = requests.get(url, timeout=CONFIG['timeout'])
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    temp_df = pd.read_csv(f, header=None)
                    first_val = str(temp_df.iloc[0, 0])
                    if not first_val.replace('.', '').replace('-', '').isdigit():
                        temp_df = temp_df.iloc[1:].reset_index(drop=True)
                    temp_df.columns = KLINE_COLUMNS[:len(temp_df.columns)]
                    kline_df = temp_df
            break
        except Exception:
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(1 * (attempt + 1))
            continue
    
    # 下载标记价格
    mark_url = f"{MARK_PRICE_URL}/{symbol}/{interval}/{filename}"
    
    for attempt in range(CONFIG['max_retries']):
        try:
            resp = requests.get(mark_url, timeout=CONFIG['timeout'])
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    temp_df = pd.read_csv(f, header=None)
                    first_val = str(temp_df.iloc[0, 0])
                    if not first_val.replace('.', '').replace('-', '').isdigit():
                        temp_df = temp_df.iloc[1:].reset_index(drop=True)
                    temp_df.columns = MARK_PRICE_COLUMNS[:len(temp_df.columns)]
                    mark_df = temp_df
            break
        except Exception:
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(1 * (attempt + 1))
            continue
    
    return kline_df, mark_df


def download_symbol(symbol: str, output_dir: Path, months: List[Tuple[int, int]], 
                    progress: ProgressTracker) -> Tuple[str, str]:
    """下载单个标的所有数据"""
    
    output_path = output_dir / f"{symbol}.parquet"
    
    # 断点续传：已存在则跳过
    if output_path.exists():
        progress.update('skipped', symbol)
        return symbol, 'skipped'
    
    all_kline_dfs = []
    all_mark_dfs = []
    
    for year, month in months:
        kline_df, mark_df = download_single_month(symbol, year, month)
        if kline_df is not None and len(kline_df) > 0:
            all_kline_dfs.append(kline_df)
        if mark_df is not None and len(mark_df) > 0:
            all_mark_dfs.append(mark_df)
        time.sleep(0.02)
    
    if not all_kline_dfs:
        progress.update('failed', f"{symbol} (no data)")
        return symbol, 'no_data'
    
    # 合并
    combined = pd.concat(all_kline_dfs, ignore_index=True)
    
    # 时间戳
    combined['open_time'] = pd.to_numeric(combined['open_time'], errors='coerce')
    if combined['open_time'].iloc[0] > 1e15:
        combined['open_time'] = pd.to_datetime(combined['open_time'], unit='us')
    else:
        combined['open_time'] = pd.to_datetime(combined['open_time'], unit='ms')
    
    # 删除不需要的列
    drop_cols = ['close_time', 'ignore']
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
    
    # 合并标记价格
    if all_mark_dfs:
        mark_combined = pd.concat(all_mark_dfs, ignore_index=True)
        mark_combined['open_time'] = pd.to_numeric(mark_combined['open_time'], errors='coerce')
        if mark_combined['open_time'].iloc[0] > 1e15:
            mark_combined['open_time'] = pd.to_datetime(mark_combined['open_time'], unit='us')
        else:
            mark_combined['open_time'] = pd.to_datetime(mark_combined['open_time'], unit='ms')
        
        mark_cols = ['open_time', 'mark_open', 'mark_high', 'mark_low', 'mark_close']
        mark_combined = mark_combined[mark_cols].drop_duplicates(subset=['open_time'])
        
        for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
            mark_combined[col] = pd.to_numeric(mark_combined[col], errors='coerce')
        
        combined = combined.merge(mark_combined, on='open_time', how='left')
    
    combined['symbol'] = symbol
    
    # 转换类型（float32节省空间）
    float_cols = ['open', 'high', 'low', 'close', 'volume', 
                  'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
    for col in float_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce').astype('float32')
    
    if 'trades' in combined.columns:
        combined['trades'] = pd.to_numeric(combined['trades'], errors='coerce').astype('int32')
    
    for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
        if col in combined.columns:
            combined[col] = combined[col].astype('float32')
    
    # 去重排序
    combined = combined.drop_duplicates(subset=['open_time']).sort_values('open_time')
    
    # 列顺序
    output_cols = ['open_time', 'symbol', 'open', 'high', 'low', 'close', 
                   'volume', 'quote_volume', 'trades', 
                   'taker_buy_volume', 'taker_buy_quote_volume']
    if 'mark_close' in combined.columns:
        output_cols.extend(['mark_open', 'mark_high', 'mark_low', 'mark_close'])
    
    combined = combined[output_cols]
    
    # 保存
    combined.to_parquet(output_path, compression='snappy', index=False)
    
    rows = len(combined)
    size_mb = output_path.stat().st_size / 1024 / 1024
    progress.update('success', f"{symbol} ({rows:,}, {size_mb:.0f}MB)")
    return symbol, 'success'


def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Binance 永续合约 1分钟K线数据下载器 (扩展版)")
    print("=" * 70)
    print(f"票池: {len(QUALIFIED_POOL)} 个标的 (流动性前90%)")
    print(f"时间: {CONFIG['start_year']}-{CONFIG['start_month']:02d} ~ {CONFIG['end_year']}-{CONFIG['end_month']:02d}")
    print(f"周期: {CONFIG['interval']}")
    print(f"并发: {CONFIG['max_workers']}")
    print(f"输出: {output_dir.absolute()}")
    print(f"格式: {{symbol}}.parquet")
    print("=" * 70)
    
    # 生成月份列表
    months = generate_month_list()
    print(f"月份数: {len(months)}")
    
    # 检查已完成
    existing = sum(1 for s in QUALIFIED_POOL if (output_dir / f"{s}.parquet").exists())
    print(f"已完成: {existing}")
    print(f"待下载: {len(QUALIFIED_POOL) - existing}")
    
    est_bars = 30 * 24 * 60 * len(months)
    print(f"预估每标的: ~{est_bars:,} bars")
    print("-" * 70)
    print("开始下载...\n")
    
    progress = ProgressTracker(len(QUALIFIED_POOL))
    results = {}
    
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(download_symbol, symbol, output_dir, months, progress): symbol
            for symbol in QUALIFIED_POOL
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, status = future.result()
                results[sym] = status
            except Exception as e:
                results[symbol] = f'error: {str(e)[:30]}'
    
    print("\n\n" + "=" * 70)
    print("下载完成!")
    print("=" * 70)
    print(f"成功: {progress.success}")
    print(f"跳过: {progress.skipped}")
    print(f"失败: {progress.failed}")
    print(f"耗时: {time.time() - progress.start_time:.1f}秒")
    
    # 统计文件
    files = list(output_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files) / 1024 / 1024 / 1024
    print(f"\n文件: {len(files)} 个 Parquet")
    print(f"大小: {total_size:.2f} GB")
    
    # 日志
    log_path = output_dir / "download_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"时间: {datetime.now().isoformat()}\n")
        f.write(f"范围: {CONFIG['start_year']}-{CONFIG['start_month']:02d} ~ {CONFIG['end_year']}-{CONFIG['end_month']:02d}\n")
        f.write(f"成功: {progress.success}\n")
        f.write(f"跳过: {progress.skipped}\n")
        f.write(f"失败: {progress.failed}\n\n")
        for symbol, status in sorted(results.items()):
            f.write(f"{symbol}: {status}\n")
    
    print(f"日志: {log_path}")
    
    if progress.failed > 0:
        print(f"\n⚠️  有 {progress.failed} 个标的失败，重新运行可继续下载")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，已下载的文件已保存，重新运行可继续")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)