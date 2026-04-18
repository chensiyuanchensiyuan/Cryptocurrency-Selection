"""
加密货币日内因子挖掘系统

从分钟级数据中提取日内特征，聚合为日频因子
输出：factors/intraday_factors.csv

数据说明：
- 数据来源：币安U本位永续合约历史K线
- 时间格式：UTC时间（Unix毫秒时间戳）
- 时段定义（基于UTC）：
  * 亚洲时段: UTC 00:00-08:00 = 北京时间 08:00-16:00 = 东京 09:00-17:00
  * 欧洲时段: UTC 07:00-16:00 = 伦敦时间 07:00-16:00 = 法兰克福 08:00-17:00
  * 美洲时段: UTC 13:00-22:00 = 纽约时间 08:00-17:00 (EST冬令时)
  * 欧美重叠: UTC 13:00-16:00 = 流动性最高时段
  * 深夜时段: UTC 22:00-01:00 = 全球流动性最低时段

因子来源（15篇研报）：
- 方正证券《成交量激增时刻蕴含的alpha信息》2022.04 - 适度冒险因子
- 方正证券《个股成交量的潮汐变化》2022.05 - 潮汐因子
- 方正证券《波动率的波动率与投资者模糊性厌恶》2022.08 - 云开雾散因子
- 方正证券《个股股价跳跃及其对振幅因子的改进》2022.09 - 飞蛾扑火因子
- 信达证券《基于分钟线的高频选股因子》2022.04 - 改进反转/动量因子
- 东吴证券《成交价改进换手率因子》2022.08 - TPS/SPS因子
- 东吴证券《优加换手率UTR选股因子2.0》2023.05 - 量稳因子
- 东吴证券《成交量对动量因子的修正》2023.06 - 聪明动量因子
- 东吴证券《反应不足or反应过度》2023.07 - 信息分布因子
- 开源证券《日内极端收益前后的反转特性》2022.12 - ERR因子
- 开源证券《大小单资金流alpha探究2.0》2022.12 - 资金流因子
- 开源证券《日内分钟收益率的时序特征》2022.12 - TGD因子
- 华安证券《交易量对波动率的非对称效应》2023.09 - 非对称波动因子
- 东方证券《因子选股系列之十六：非对称价格冲击》2016.11 - gammabias因子
- 东方证券《因子选股系列之十四：流动性度量》2016.11 - Lambda/Amihud ILLIQ因子

加密市场适配：
- 时段因子：使用亚洲/欧洲/美洲UTC时段划分
- Taker资金流因子：利用加密市场独有的taker_buy数据
- 流动性因子：24/7市场的分时段流动性分析

使用方法:
    python factor_mining.py
    python factor_mining.py --data_dir ./futures_data_1m

因子筛选请运行: python factor_screening.py
"""

import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============== 配置 ==============
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

# ============== 因子注册 ==============
@dataclass
class FactorMeta:
    name: str
    formula: str
    description: str
    category: str = "intraday"
    source: str = ""  # 研报来源

FACTOR_LIBRARY: Dict[str, FactorMeta] = {}

def register_factor(name: str, formula: str, description: str, category: str = "intraday", source: str = ""):
    if name not in FACTOR_LIBRARY:
        FACTOR_LIBRARY[name] = FactorMeta(name=name, formula=formula, 
                                          description=description, category=category, source=source)


# ============================================================
# 数据加载
# ============================================================

def load_minute_data_and_resample(data_dir: str) -> tuple:
    """
    加载分钟数据，同时返回:
    1. 用于日内因子计算的分钟数据（按symbol分块）
    2. 聚合后的日频数据（用于计算fwd_ret和因子分析）
    """
    print(f"Loading minute data from {data_dir}...")
    
    data_path = Path(data_dir)
    daily_dfs = []
    loaded = 0
    skipped = 0
    
    # 存储每个symbol的文件路径（用于后续逐个处理）
    symbol_files = {}
    
    for symbol in QUALIFIED_POOL:
        file_path = data_path / f"{symbol}.parquet"
        if not file_path.exists():
            skipped += 1
            continue
        
        symbol_files[symbol] = file_path
        
        # 加载并聚合为日频
        df = pd.read_parquet(file_path)
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['date'] = df['open_time'].dt.date
        
        # 聚合为日频
        agg_dict = {
            'symbol': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
            'trades': 'sum',
            'taker_buy_volume': 'sum',
        }
        
        if 'mark_close' in df.columns:
            agg_dict.update({
                'mark_open': 'first',
                'mark_high': 'max',
                'mark_low': 'min',
                'mark_close': 'last',
            })
        
        daily = df.groupby('date').agg(agg_dict).reset_index()
        daily_dfs.append(daily)
        loaded += 1
        
        del df
        gc.collect()
        
        if loaded % 50 == 0:
            print(f"  Loaded {loaded} symbols...")
    
    if not daily_dfs:
        raise ValueError(f"No data found in {data_dir}")
    
    daily_df = pd.concat(daily_dfs, ignore_index=True)
    daily_df = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"Loaded {loaded} symbols, skipped {skipped}")
    print(f"Daily data: {len(daily_df):,} rows")
    print(f"Date range: {daily_df['date'].min()} ~ {daily_df['date'].max()}")
    
    return symbol_files, daily_df


# ============================================================
# 日内因子计算 - 原始因子
# ============================================================

def compute_base_factors(group: pd.DataFrame, symbol: str, date, price_col: str) -> Optional[Dict]:
    """计算基础日内因子（原有因子）"""
    result = {'symbol': symbol, 'date': date}
    
    # ret第一条可能是NaN（symbol首日），fillna(0)保持与volumes/hours对齐
    ret = group['ret'].fillna(0).values
    if len(ret) < 30:
        return None
    
    prices = group[price_col].values
    volumes = group['volume'].values if 'volume' in group.columns else None
    hours = group['hour'].values
    
    # ========== 【1】收益率分布因子 ==========
    register_factor('realized_skew', 'skew(ret_1m)', 
                    '分钟收益率偏度，负偏度预示下跌风险，正偏度预示上涨潜力', source='基础因子')
    result['realized_skew'] = float(stats.skew(ret))
    
    register_factor('realized_kurt', 'kurtosis(ret_1m)', 
                    '分钟收益率峰度，高峰度表示极端收益频繁，尾部风险大', source='基础因子')
    result['realized_kurt'] = float(stats.kurtosis(ret))
    
    register_factor('realized_vol', 'sqrt(sum(ret^2))', 
                    '已实现波动率，衡量日内价格波动剧烈程度', source='基础因子')
    result['realized_vol'] = float(np.sqrt((ret ** 2).sum()))
    
    # 上下行波动分解
    up_ret, down_ret = ret[ret > 0], ret[ret < 0]
    up_vol = np.sqrt((up_ret ** 2).sum()) if len(up_ret) > 0 else 0
    down_vol = np.sqrt((down_ret ** 2).sum()) if len(down_ret) > 0 else 0
    total_vol = up_vol + down_vol
    
    register_factor('down_vol_ratio', 'down_vol/(up_vol+down_vol)', 
                    '下行波动占比，高值表示下跌风险主导，低值表示上涨动能强', source='基础因子')
    result['down_vol_ratio'] = down_vol / total_vol if total_vol > 0 else np.nan
    
    register_factor('vol_asymmetry', '(up_vol-down_vol)/(up_vol+down_vol)', 
                    '波动不对称性，正值表示上涨波动大于下跌波动', source='基础因子')
    result['vol_asymmetry'] = (up_vol - down_vol) / total_vol if total_vol > 0 else np.nan
    
    # ========== 【2】日内动量/反转因子 ==========
    n = len(ret)
    mid = n // 2
    first_half = ret[:mid].sum()
    second_half = ret[mid:].sum()
    
    register_factor('first_half_ret', 'sum(ret[0:N/2])', 
                    '日内前半段累计收益(UTC日内前12小时)，反映亚欧时段市场情绪', source='基础因子')
    result['first_half_ret'] = float(first_half)
    
    register_factor('second_half_ret', 'sum(ret[N/2:N])', 
                    '日内后半段累计收益(UTC日内后12小时)，反映美洲时段市场情绪', source='基础因子')
    result['second_half_ret'] = float(second_half)
    
    register_factor('intraday_reversal', '-first_half*sign(second_half)', 
                    '日内反转因子，捕捉日内价格均值回归效应', source='基础因子')
    result['intraday_reversal'] = -first_half * np.sign(second_half) if second_half != 0 else 0
    
    # 日内首尾60分钟（注：基于UTC日切换，即UTC 00:00后的首60分钟）
    first_hour = ret[:60] if len(ret) >= 60 else ret
    last_hour = ret[-60:] if len(ret) >= 60 else ret
    
    register_factor('first_hour_ret', 'sum(ret[0:60])', 
                    '日内首60分钟收益(UTC 00:00后)，对应北京时间08:00-09:00', source='基础因子')
    result['first_hour_ret'] = float(first_hour.sum())
    
    register_factor('last_hour_ret', 'sum(ret[-60:])', 
                    '日内末60分钟收益(UTC 23:00后)，对应北京时间07:00-08:00(次日)', source='基础因子')
    result['last_hour_ret'] = float(last_hour.sum())
    
    # 日内首30分钟vs后续时段（基于UTC日切换）
    if len(ret) >= 60:
        early_ret = ret[:30].sum()
        main_ret = ret[30:].sum()
        register_factor('early_main_diff', 'ret[0:30] - ret[30:]', 
                        'UTC日内首30分钟vs后续时段收益差，反映日内动量/反转', source='基础因子')
        result['early_main_diff'] = float(early_ret - main_ret)
    
    # ========== 【3】时段效应因子 ==========
    # 币安数据为UTC时间，各时段对应关系：
    # - 亚洲时段 UTC 00:00-08:00 = 北京时间 08:00-16:00 (东京09:00-17:00, 香港08:00-16:00)
    # - 欧洲时段 UTC 07:00-16:00 = 伦敦时间 07:00-16:00 (法兰克福08:00-17:00)
    # - 美洲时段 UTC 13:00-22:00 = 纽约时间 08:00-17:00 (EST冬令时)
    # - 深夜低流动性时段 UTC 22:00-01:00 = 全球交易最不活跃时段
    ret_len = len(ret)
    if len(hours) >= ret_len:
        hours_aligned = hours[:ret_len]
        
        # 亚洲时段 (UTC 00:00-08:00) = 北京08:00-16:00
        asia_mask = (hours_aligned >= 0) & (hours_aligned < 8)
        asia_ret = ret[asia_mask]
        register_factor('asia_session_ret', 'sum(ret[UTC 0-8])', 
                        '亚洲时段收益(UTC 0-8=北京08-16时)，反映亚洲市场情绪', source='基础因子')
        result['asia_session_ret'] = float(asia_ret.sum()) if len(asia_ret) > 0 else np.nan
        
        # 欧洲时段 (UTC 07:00-16:00) = 伦敦07:00-16:00
        eu_mask = (hours_aligned >= 7) & (hours_aligned < 16)
        eu_ret = ret[eu_mask]
        register_factor('eu_session_ret', 'sum(ret[UTC 7-16])', 
                        '欧洲时段收益(UTC 7-16=伦敦07-16时)，反映欧洲市场情绪', source='基础因子')
        result['eu_session_ret'] = float(eu_ret.sum()) if len(eu_ret) > 0 else np.nan
        
        # 美洲时段 (UTC 13:00-22:00) = 纽约08:00-17:00 EST
        us_mask = (hours_aligned >= 13) & (hours_aligned < 22)
        us_ret = ret[us_mask]
        register_factor('us_session_ret', 'sum(ret[UTC 13-22])', 
                        '美洲时段收益(UTC 13-22=纽约08-17时EST)，反映美国市场情绪', source='基础因子')
        result['us_session_ret'] = float(us_ret.sum()) if len(us_ret) > 0 else np.nan
        
        # 欧美重叠时段 (UTC 13:00-16:00) = 伦敦13-16时 + 纽约08-11时
        overlap_mask = (hours_aligned >= 13) & (hours_aligned < 16)
        overlap_ret = ret[overlap_mask]
        register_factor('eu_us_overlap_ret', 'sum(ret[UTC 13-16])', 
                        '欧美重叠时段收益(UTC 13-16)，流动性最高时段', source='基础因子')
        result['eu_us_overlap_ret'] = float(overlap_ret.sum()) if len(overlap_ret) > 0 else np.nan
        
        # 深夜低流动性时段 (UTC 22:00-01:00) = 全球交易最少时段
        # 注意：跨日处理
        quiet_mask = (hours_aligned >= 22) | (hours_aligned < 1)
        quiet_ret = ret[quiet_mask]
        register_factor('quiet_hours_ret', 'sum(ret[UTC 22-01])', 
                        '深夜时段收益(UTC 22-01)，全球流动性最低时段', source='基础因子')
        result['quiet_hours_ret'] = float(quiet_ret.sum()) if len(quiet_ret) > 0 else np.nan
    
    # ========== 【4】成交量分布因子 ==========
    if volumes is not None and len(volumes) > 0:
        total_vol_sum = volumes.sum()
        
        if total_vol_sum > 0:
            # 尾盘成交占比
            tail_vol = volumes[-30:].sum() if len(volumes) >= 30 else volumes.sum()
            register_factor('tail_vol_ratio', 'sum(vol[-30:])/sum(vol)', 
                            '尾盘30分钟成交占比，高值可能预示次日延续趋势', source='基础因子')
            result['tail_vol_ratio'] = float(tail_vol / total_vol_sum)
            
            # 早盘成交占比
            head_vol = volumes[:60].sum() if len(volumes) >= 60 else volumes.sum()
            register_factor('head_vol_ratio', 'sum(vol[:60])/sum(vol)', 
                            '早盘60分钟成交占比，高值表示开盘活跃', source='基础因子')
            result['head_vol_ratio'] = float(head_vol / total_vol_sum)
            
            # 成交量集中度
            hourly_vol = np.array([volumes[hours == h].sum() for h in range(24)])
            hourly_vol = hourly_vol[hourly_vol > 0]
            if len(hourly_vol) > 0:
                register_factor('vol_concentration', 'max(hourly_vol)/mean(hourly_vol)', 
                                '成交量小时集中度，高值表示成交集中在特定时段', source='基础因子')
                result['vol_concentration'] = float(hourly_vol.max() / hourly_vol.mean())
        
        # 日内量价相关性
        if len(ret) > 30:
            vol_for_corr = volumes[1:len(ret)+1] if len(volumes) > len(ret) else volumes[:len(ret)]
            if len(vol_for_corr) >= len(ret):
                vol_for_corr = vol_for_corr[:len(ret)]
                valid = ~(np.isnan(ret) | np.isnan(vol_for_corr))
                if valid.sum() > 30:
                    rv_corr = np.corrcoef(ret[valid], vol_for_corr[valid])[0, 1]
                    register_factor('intraday_rv_corr', 'corr(ret, vol)', 
                                    '日内量价相关性，正相关表示放量上涨', source='基础因子')
                    result['intraday_rv_corr'] = float(rv_corr) if not np.isnan(rv_corr) else np.nan
    
    # ========== 【5】Taker资金流因子 ==========
    if 'taker_imb' in group.columns:
        taker_imb = group['taker_imb'].dropna().values
        
        if len(taker_imb) > 10:
            register_factor('taker_imbalance_mean', 'mean(taker_imb)', 
                            'Taker净买入比例均值，正值表示主动买入占优', source='基础因子')
            result['taker_imbalance_mean'] = float(taker_imb.mean())
            
            register_factor('taker_imbalance_std', 'std(taker_imb)', 
                            'Taker不平衡波动，高值表示买卖力量频繁转换', source='基础因子')
            result['taker_imbalance_std'] = float(taker_imb.std())
            
            # Taker持续性（自相关）
            if len(taker_imb) > 10:
                autocorr = np.corrcoef(taker_imb[:-1], taker_imb[1:])[0, 1]
                register_factor('taker_persistence', 'autocorr(taker_imb)', 
                                'Taker资金流持续性，高值表示资金流方向稳定', source='基础因子')
                result['taker_persistence'] = float(autocorr) if not np.isnan(autocorr) else np.nan
            
            # 滚动买卖压力
            if len(taker_imb) >= 60:
                rolling = np.convolve(taker_imb, np.ones(60), mode='valid')
                register_factor('max_buy_pressure_1h', 'max(rolling_sum(taker,60))', 
                                '1小时最大买压，反映日内最强买入时刻', source='基础因子')
                result['max_buy_pressure_1h'] = float(rolling.max())
                
                register_factor('max_sell_pressure_1h', 'min(rolling_sum(taker,60))', 
                                '1小时最大卖压，反映日内最强卖出时刻', source='基础因子')
                result['max_sell_pressure_1h'] = float(rolling.min())
            
            # 尾盘Taker方向
            tail_taker = taker_imb[-30:] if len(taker_imb) >= 30 else taker_imb
            register_factor('tail_taker_imb', 'mean(taker_imb[-30:])', 
                            '尾盘Taker净买入，反映收盘前资金流向', source='基础因子')
            result['tail_taker_imb'] = float(tail_taker.mean())
    
    # ========== 【6】微观结构因子 ==========
    if 'avg_trade_size' in group.columns:
        trade_size = group['avg_trade_size'].dropna().values
        trade_size = trade_size[trade_size > 0]
        
        if len(trade_size) > 10:
            register_factor('avg_trade_size_mean', 'mean(vol/trades)', 
                            '平均单笔成交量，高值可能表示机构参与', source='基础因子')
            result['avg_trade_size_mean'] = float(trade_size.mean())
            
            register_factor('trade_size_skew', 'skew(trade_size)', 
                            '单笔成交量偏度，正偏表示偶发大单', source='基础因子')
            result['trade_size_skew'] = float(stats.skew(trade_size))
            
            # 大小单分析
            q70 = np.percentile(trade_size, 70)
            q30 = np.percentile(trade_size, 30)
            
            if len(ret) >= len(trade_size):
                ret_aligned = ret[:len(trade_size)]
                big_mask = trade_size > q70
                small_mask = trade_size < q30
                
                if big_mask.sum() > 5 and small_mask.sum() > 5:
                    big_ret = ret_aligned[big_mask].sum()
                    small_ret = ret_aligned[small_mask].sum()
                    
                    register_factor('big_small_ret_diff', 'big_trade_ret - small_trade_ret', 
                                    '大小单收益差，正值表示大单推动上涨', source='基础因子')
                    result['big_small_ret_diff'] = float(big_ret - small_ret)
    
    # ========== 【7】价格模式因子 ==========
    if len(prices) > 10:
        cum_max = np.maximum.accumulate(prices)
        cum_min = np.minimum.accumulate(prices)
        
        higher_high = np.sum(np.diff(cum_max) > 0)
        lower_low = np.sum(np.diff(cum_min) < 0)
        
        register_factor('trend_strength_intra', '(hh_count-ll_count)/N', 
                        '日内趋势强度，正值表示上涨趋势，负值表示下跌趋势', source='基础因子')
        result['trend_strength_intra'] = float((higher_high - lower_low) / len(prices))
        
        register_factor('intraday_range', '(max-min)/open', 
                        '日内振幅，高值表示波动剧烈', source='基础因子')
        result['intraday_range'] = float((prices.max() - prices.min()) / prices[0]) if prices[0] > 0 else np.nan
        
        # 日内最大回撤
        drawdown = (prices - cum_max) / (cum_max + 1e-8)
        register_factor('max_drawdown_intra', 'min((p-cummax)/cummax)', 
                        '日内最大回撤，反映日内下行风险', source='基础因子')
        result['max_drawdown_intra'] = float(drawdown.min())
        
        # 收盘位置
        register_factor('close_position_intra', '(close-low)/(high-low)', 
                        '日内收盘位置，高值表示收在日内高点附近', source='基础因子')
        high_low_range = prices.max() - prices.min()
        if high_low_range > 0:
            result['close_position_intra'] = float((prices[-1] - prices.min()) / high_low_range)
        else:
            result['close_position_intra'] = np.nan
    
    # ========== 【8】极值因子 ==========
    if len(ret) >= 60:
        rolling_1h = np.convolve(ret, np.ones(60), mode='valid')
        
        register_factor('max_1h_ret', 'max(rolling_sum(ret,60))', 
                        '最大1小时收益，反映日内最强上涨动能', source='基础因子')
        result['max_1h_ret'] = float(rolling_1h.max())
        
        register_factor('min_1h_ret', 'min(rolling_sum(ret,60))', 
                        '最小1小时收益，反映日内最强下跌动能', source='基础因子')
        result['min_1h_ret'] = float(rolling_1h.min())
        
        register_factor('ret_range_1h', 'max_1h - min_1h', 
                        '1小时收益极差，反映日内波动幅度', source='基础因子')
        result['ret_range_1h'] = float(rolling_1h.max() - rolling_1h.min())
    
    # ========== 【9】收益率序列特征 ==========
    if len(ret) >= 30:
        # 收益率自相关
        autocorr_1 = np.corrcoef(ret[:-1], ret[1:])[0, 1]
        register_factor('ret_autocorr_1', 'autocorr(ret, lag=1)', 
                        '收益率1阶自相关，正值表示趋势延续，负值表示反转', source='基础因子')
        result['ret_autocorr_1'] = float(autocorr_1) if not np.isnan(autocorr_1) else np.nan
        
        # 正负收益比
        pos_count = (ret > 0).sum()
        neg_count = (ret < 0).sum()
        register_factor('pos_neg_ratio', 'count(ret>0)/count(ret<0)', 
                        '正负收益比，高值表示上涨分钟数多于下跌', source='基础因子')
        result['pos_neg_ratio'] = float(pos_count / (neg_count + 1)) if neg_count > 0 else np.nan
        
        # 大幅波动占比
        ret_std = ret.std()
        large_move = (np.abs(ret) > 2 * ret_std).sum() / len(ret)
        register_factor('large_move_ratio', 'count(|ret|>2*std)/N', 
                        '大幅波动占比，高值表示极端波动频繁', source='基础因子')
        result['large_move_ratio'] = float(large_move)
    
    return result


# ============================================================
# 日内因子计算 - 新增因子（研报来源）
# ============================================================

def compute_enhanced_factors(group: pd.DataFrame, symbol: str, date, price_col: str,
                             btc_ret: Optional[np.ndarray] = None) -> Dict:
    """计算增强因子（来自研报）

    Args:
        group: 单日分钟级数据
        symbol: 币种符号
        date: 日期
        price_col: 价格列名
        btc_ret: BTC分钟收益率数组（用于市场回归计算残差），可选
    """
    result = {}
    
    # ret第一条可能是NaN（symbol首日），fillna(0)保持与volumes/hours对齐
    ret = group['ret'].fillna(0).values
    if len(ret) < 60:
        return result
    
    prices = group[price_col].values
    volumes = group['volume'].values if 'volume' in group.columns else None
    hours = group['hour'].values if 'hour' in group.columns else None
    
    n = len(ret)
    
    # ========== 【10】方正-成交量激增因子 (适度冒险) ==========
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        mean_vol = vol_aligned.mean()
        std_vol = vol_aligned.std()
        
        if std_vol > 0:
            # 激增时刻：成交量超过均值+2倍标准差
            surge_threshold = mean_vol + 2 * std_vol
            surge_mask = vol_aligned > surge_threshold
            
            if surge_mask.sum() >= 3:
                # 激增时刻的收益率
                surge_ret = ret[surge_mask]
                non_surge_ret = ret[~surge_mask]
                
                register_factor('surge_ret_mean', 'mean(ret[vol_surge])', 
                                '成交量激增时刻收益均值，方正-适度冒险因子', source='方正证券2022.04')
                result['surge_ret_mean'] = float(surge_ret.mean())
                
                register_factor('surge_vol_ratio', 'mean(ret[surge])/std(ret[surge])', 
                                '激增时刻收益波动比，衡量激增时刻的风险调整收益', source='方正证券2022.04')
                if surge_ret.std() > 0:
                    result['surge_vol_ratio'] = float(surge_ret.mean() / surge_ret.std())
                
                # 耀眼波动率：激增时刻波动vs日内平均波动
                surge_volatility = np.abs(surge_ret).mean()
                avg_volatility = np.abs(ret).mean()
                register_factor('brilliant_vol', 'vol_surge / vol_avg', 
                                '耀眼波动率，激增时刻波动相对日内平均波动', source='方正证券2022.04')
                if avg_volatility > 0:
                    result['brilliant_vol'] = float(surge_volatility / avg_volatility)
                
                # 耀眼收益率：激增时刻收益vs日内平均
                register_factor('brilliant_ret', 'ret_surge / ret_avg', 
                                '耀眼收益率，激增时刻收益相对日内平均', source='方正证券2022.04')
                if non_surge_ret.mean() != 0:
                    result['brilliant_ret'] = float(surge_ret.mean() / (np.abs(non_surge_ret.mean()) + 1e-8))
    
    # ========== 【11】方正-潮汐因子 ==========
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        
        # 计算累积成交量占比
        cum_vol = np.cumsum(vol_aligned)
        total_vol = cum_vol[-1]
        
        if total_vol > 0:
            vol_pct = cum_vol / total_vol
            
            # 找到成交量达到50%的时点（潮汐高点）
            half_idx = np.searchsorted(vol_pct, 0.5)
            
            # 潮汐因子：成交量集中在前半段还是后半段
            register_factor('tide_position', 'idx(vol_cum=0.5) / N', 
                            '潮汐位置，<0.5表示成交前移，>0.5表示成交后移', source='方正证券2022.05')
            result['tide_position'] = float(half_idx / n)
            
            # 强势半潮汐：上半段涨幅时的成交量占比
            mid = n // 2
            first_half_vol = vol_aligned[:mid].sum()
            first_half_ret = ret[:mid].sum()
            
            register_factor('strong_tide', 'vol_first_half * sign(ret_first_half)', 
                            '强势半潮汐，正值表示上涨时成交活跃', source='方正证券2022.05')
            result['strong_tide'] = float((first_half_vol / total_vol) * np.sign(first_half_ret))
            
            # 潮汐强度：前后半段成交量差异
            second_half_vol = vol_aligned[mid:].sum()
            register_factor('tide_intensity', '(vol_first - vol_second) / total_vol', 
                            '潮汐强度，正值表示成交前移', source='方正证券2022.05')
            result['tide_intensity'] = float((first_half_vol - second_half_vol) / total_vol)
    
    # ========== 【12】方正-波动率模糊性因子 (云开雾散) ==========
    if len(ret) >= 60:
        # 计算滚动波动率的波动率（模糊性）
        window = 30
        rolling_vol = np.array([ret[i:i+window].std() for i in range(n - window + 1)])
        
        if len(rolling_vol) > 10:
            vol_of_vol = np.std(rolling_vol)
            
            register_factor('vol_of_vol', 'std(rolling_vol)', 
                            '波动率的波动率（模糊性），高值表示波动不稳定', source='方正证券2022.08')
            result['vol_of_vol'] = float(vol_of_vol)
            
            # 模糊性与成交量的关系
            if volumes is not None and len(volumes) >= n:
                vol_aligned = volumes[window-1:n]
                if len(vol_aligned) == len(rolling_vol):
                    valid = ~(np.isnan(rolling_vol) | np.isnan(vol_aligned))
                    if valid.sum() > 20:
                        fuzzy_corr = np.corrcoef(rolling_vol[valid], vol_aligned[valid])[0, 1]
                        register_factor('fuzzy_vol_corr', 'corr(vol_of_vol, volume)', 
                                        '模糊性与成交量相关性，反映模糊时的交易行为', source='方正证券2022.08')
                        if not np.isnan(fuzzy_corr):
                            result['fuzzy_vol_corr'] = float(fuzzy_corr)
            
            # 高模糊时刻的收益
            high_fuzzy_mask = rolling_vol > np.percentile(rolling_vol, 80)
            if high_fuzzy_mask.sum() > 5:
                fuzzy_ret = ret[window-1:n][high_fuzzy_mask]
                register_factor('fuzzy_ret_mean', 'mean(ret[high_fuzzy])', 
                                '高模糊时刻收益，云开雾散因子', source='方正证券2022.08')
                result['fuzzy_ret_mean'] = float(fuzzy_ret.mean())
    
    # ========== 【13】方正-股价跳跃因子 (飞蛾扑火) ==========
    if len(ret) >= 30:
        # 跳跃度：使用单利和连续复利的差异来衡量
        # 简化版：使用收益率平方和的差异
        simple_ret = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        compound_ret = ret.sum()
        
        # 跳跃度 = |简单收益 - 复合收益|
        register_factor('jump_intensity', 'abs(simple_ret - compound_ret)', 
                        '跳跃度，反映价格非连续变化程度', source='方正证券2022.09')
        result['jump_intensity'] = float(np.abs(simple_ret - compound_ret))
        
        # 使用分钟收益率的极端值来识别跳跃
        ret_std = ret.std()
        if ret_std > 0:
            extreme_mask = np.abs(ret) > 3 * ret_std
            jump_count = extreme_mask.sum()
            
            register_factor('jump_count', 'count(|ret| > 3*std)', 
                            '跳跃次数，反映日内极端波动频率', source='方正证券2022.09')
            result['jump_count'] = float(jump_count)
            
            # 跳跃贡献度：极端收益对总收益的贡献
            if extreme_mask.sum() > 0:
                jump_contribution = np.abs(ret[extreme_mask]).sum() / (np.abs(ret).sum() + 1e-8)
                register_factor('jump_contribution', 'sum(|ret_jump|) / sum(|ret|)', 
                                '跳跃贡献度，反映极端波动对总波动的贡献', source='方正证券2022.09')
                result['jump_contribution'] = float(jump_contribution)
    
    # ========== 【14】开源-极端收益因子 (ERR) ==========
    if len(ret) >= 30:
        # 找到最极端收益（偏离中位数最远）
        median_ret = np.median(ret)
        deviation = np.abs(ret - median_ret)
        extreme_idx = np.argmax(deviation)
        
        # 最极端收益
        extreme_ret = ret[extreme_idx]
        register_factor('extreme_ret', 'ret[argmax(|ret-median|)]', 
                        '最极端收益，日内偏离中位数最大的收益', source='开源证券2022.12')
        result['extreme_ret'] = float(extreme_ret)
        
        # 极端收益前的收益（反转信号）
        if extreme_idx > 0:
            pre_extreme_ret = ret[extreme_idx - 1]
            register_factor('pre_extreme_ret', 'ret[extreme_idx - 1]', 
                            '极端收益前1分钟收益，ERR因子组成部分', source='开源证券2022.12')
            result['pre_extreme_ret'] = float(pre_extreme_ret)
            
            # ERR因子：极端收益 + 前1分钟收益的rank和
            register_factor('err_factor', 'extreme_ret + pre_extreme_ret', 
                            'ERR因子，极端收益与前1分钟收益之和', source='开源证券2022.12')
            result['err_factor'] = float(extreme_ret + pre_extreme_ret)
        
        # 极端收益后的收益（动量信号）
        if extreme_idx < n - 1:
            post_extreme_ret = ret[extreme_idx + 1:].sum()
            register_factor('post_extreme_ret', 'sum(ret[extreme_idx+1:])', 
                            '极端收益后累计收益，反映极端事件后的市场反应', source='开源证券2022.12')
            result['post_extreme_ret'] = float(post_extreme_ret)
    
    # ========== 【15】开源-时间重心偏离因子 (TGD) ==========
    if len(ret) >= 30:
        # 计算涨幅时间重心和跌幅时间重心
        time_idx = np.arange(n)
        
        up_mask = ret > 0
        down_mask = ret < 0
        
        if up_mask.sum() > 5 and down_mask.sum() > 5:
            # 涨幅时间重心
            up_center = (time_idx[up_mask] * np.abs(ret[up_mask])).sum() / (np.abs(ret[up_mask]).sum() + 1e-8)
            
            # 跌幅时间重心
            down_center = (time_idx[down_mask] * np.abs(ret[down_mask])).sum() / (np.abs(ret[down_mask]).sum() + 1e-8)
            
            register_factor('up_time_center', 'weighted_mean(time, |ret_up|)', 
                            '涨幅时间重心，值越大表示上涨越集中在尾盘', source='开源证券2022.12')
            result['up_time_center'] = float(up_center / n)
            
            register_factor('down_time_center', 'weighted_mean(time, |ret_down|)', 
                            '跌幅时间重心，值越大表示下跌越集中在尾盘', source='开源证券2022.12')
            result['down_time_center'] = float(down_center / n)
            
            # 时间重心偏离
            register_factor('time_center_diff', 'down_center - up_center', 
                            '时间重心偏离(TGD)，正值表示下跌偏尾盘，上涨偏早盘', source='开源证券2022.12')
            result['time_center_diff'] = float((down_center - up_center) / n)
    
    # ========== 【16】华安-非对称波动因子 ==========
    if len(ret) >= 30:
        # 正半方差和负半方差
        up_ret = ret[ret > 0]
        down_ret = ret[ret < 0]
        
        rsv_plus = (up_ret ** 2).sum() if len(up_ret) > 0 else 0
        rsv_minus = (down_ret ** 2).sum() if len(down_ret) > 0 else 0
        
        register_factor('rsv_plus', 'sum(ret_up^2)', 
                        '正半方差，衡量上涨波动贡献', source='华安证券2023.09')
        result['rsv_plus'] = float(rsv_plus)
        
        register_factor('rsv_minus', 'sum(ret_down^2)', 
                        '负半方差，衡量下跌波动贡献', source='华安证券2023.09')
        result['rsv_minus'] = float(rsv_minus)
        
        # 非对称性：负半方差/总方差
        total_var = rsv_plus + rsv_minus
        if total_var > 0:
            register_factor('rsv_asymmetry', 'rsv_minus / (rsv_plus + rsv_minus)', 
                            '波动非对称性，高值表示下跌波动主导', source='华安证券2023.09')
            result['rsv_asymmetry'] = float(rsv_minus / total_var)
        
        # 非对称交易量因子
        if volumes is not None and len(volumes) >= n:
            vol_aligned = volumes[:n]
            up_vol = vol_aligned[ret > 0].sum() if (ret > 0).sum() > 0 else 0
            down_vol = vol_aligned[ret < 0].sum() if (ret < 0).sum() > 0 else 0
            total_vol_sum = up_vol + down_vol
            
            if total_vol_sum > 0:
                register_factor('asym_volume', '(up_vol - down_vol) / total_vol', 
                                '非对称交易量，正值表示上涨时成交量大', source='华安证券2023.09')
                result['asym_volume'] = float((up_vol - down_vol) / total_vol_sum)
    
    # ========== 【17】信达-改进反转/动量因子 ==========
    if len(ret) >= 30 and volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        
        # 正收益分钟和负收益分钟分开计算
        up_mask = ret > 0
        down_mask = ret < 0
        
        # 高成交量时的正收益反转
        vol_median = np.median(vol_aligned)
        high_vol_mask = vol_aligned > vol_median
        
        if (up_mask & high_vol_mask).sum() > 5:
            high_vol_up_ret = ret[up_mask & high_vol_mask].sum()
            register_factor('high_vol_up_reversal', '-sum(ret_up[high_vol])', 
                            '高成交量正收益反转因子', source='信达证券2022.04')
            result['high_vol_up_reversal'] = float(-high_vol_up_ret)
        
        # 高成交量时的负收益动量
        if (down_mask & high_vol_mask).sum() > 5:
            high_vol_down_ret = ret[down_mask & high_vol_mask].sum()
            register_factor('high_vol_down_momentum', 'sum(ret_down[high_vol])', 
                            '高成交量负收益动量因子', source='信达证券2022.04')
            result['high_vol_down_momentum'] = float(high_vol_down_ret)
        
        # 改进波动率因子：高成交量时的波动率
        if high_vol_mask.sum() > 10:
            high_vol_volatility = ret[high_vol_mask].std()
            low_vol_volatility = ret[~high_vol_mask].std() if (~high_vol_mask).sum() > 10 else 0
            
            if low_vol_volatility > 0:
                register_factor('vol_weighted_volatility', 'std(ret[high_vol]) / std(ret[low_vol])', 
                                '成交量加权波动率比，衡量高成交量时的相对波动', source='信达证券2022.04')
                result['vol_weighted_volatility'] = float(high_vol_volatility / low_vol_volatility)
    
    # ========== 【18】东吴-聪明动量因子 ==========
    if len(ret) >= 60 and volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        
        # 计算"聪明"指标：每分钟标准化涨跌幅
        ret_std = ret.std()
        if ret_std > 0:
            normalized_ret = ret / ret_std
            
            # 找出波动最大的20%分钟（知情交易集中时段）
            smart_threshold = np.percentile(np.abs(normalized_ret), 80)
            smart_mask = np.abs(normalized_ret) >= smart_threshold
            
            if smart_mask.sum() >= 5:
                smart_ret = ret[smart_mask].sum()
                non_smart_ret = ret[~smart_mask].sum()
                
                register_factor('smart_ret', 'sum(ret[smart_minutes])', 
                                '聪明时段收益，知情交易最集中时段的收益', source='东吴证券2023.06')
                result['smart_ret'] = float(smart_ret)
                
                register_factor('smart_non_smart_diff', 'smart_ret - non_smart_ret', 
                                '聪明与非聪明时段收益差', source='东吴证券2023.06')
                result['smart_non_smart_diff'] = float(smart_ret - non_smart_ret)
                
                # 聪明时段的成交量占比
                smart_vol = vol_aligned[smart_mask].sum()
                total_vol_sum = vol_aligned.sum()
                if total_vol_sum > 0:
                    register_factor('smart_vol_ratio', 'sum(vol[smart]) / total_vol', 
                                    '聪明时段成交量占比', source='东吴证券2023.06')
                    result['smart_vol_ratio'] = float(smart_vol / total_vol_sum)
    
    # ========== 【19】东吴-信息分布因子 ==========
    if len(ret) >= 60:
        # 信息分布均匀度：使用收益率的时间分布熵
        # 将一天分成若干时段，计算每个时段收益率绝对值占比
        n_periods = 12  # 每2小时一个时段
        period_len = n // n_periods
        
        period_abs_ret = []
        for i in range(n_periods):
            start = i * period_len
            end = start + period_len if i < n_periods - 1 else n
            period_abs_ret.append(np.abs(ret[start:end]).sum())
        
        period_abs_ret = np.array(period_abs_ret)
        total_abs = period_abs_ret.sum()
        
        if total_abs > 0:
            # 计算信息分布熵
            prob = period_abs_ret / total_abs
            prob = prob[prob > 0]  # 避免log(0)
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            max_entropy = np.log(n_periods)
            
            register_factor('info_entropy', '-sum(p * log(p))', 
                            '信息分布熵，高值表示信息分布均匀', source='东吴证券2023.07')
            result['info_entropy'] = float(entropy)
            
            # 信息均匀度（归一化）
            register_factor('info_uniformity', 'entropy / max_entropy', 
                            '信息均匀度，越接近1表示分布越均匀', source='东吴证券2023.07')
            result['info_uniformity'] = float(entropy / max_entropy)
            
            # 信息集中度：最大时段占比
            register_factor('info_concentration', 'max(period_abs_ret) / total_abs', 
                            '信息集中度，高值表示信息集中在某时段', source='东吴证券2023.07')
            result['info_concentration'] = float(period_abs_ret.max() / total_abs)
    
    # ========== 【20】东吴-量稳因子 ==========
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        
        # 成交量稳定性：成交量的变异系数
        vol_mean = vol_aligned.mean()
        vol_std = vol_aligned.std()
        
        if vol_mean > 0:
            register_factor('vol_cv', 'std(vol) / mean(vol)', 
                            '成交量变异系数，低值表示成交量稳定', source='东吴证券2023.05')
            result['vol_cv'] = float(vol_std / vol_mean)
            
            # 量稳因子：取负数使得稳定性高的股票因子值高
            register_factor('vol_stability', '-std(vol) / mean(vol)', 
                            '量稳因子，高值表示成交量更稳定', source='东吴证券2023.05')
            result['vol_stability'] = float(-vol_std / vol_mean)
        
        # 分时段成交量稳定性
        if hours is not None and len(hours) >= n:
            hourly_vol = [vol_aligned[hours[:n] == h].sum() for h in range(24)]
            hourly_vol = np.array([v for v in hourly_vol if v > 0])
            
            if len(hourly_vol) > 5:
                hourly_cv = hourly_vol.std() / (hourly_vol.mean() + 1e-8)
                register_factor('hourly_vol_stability', '-std(hourly_vol) / mean(hourly_vol)', 
                                '小时成交量稳定性', source='东吴证券2023.05')
                result['hourly_vol_stability'] = float(-hourly_cv)
    
    # ========== 【21】收益率动量持续性 ==========
    # 注：原"水中行舟"因子需要同时刻的市场（BTC）收益率，当前数据结构不支持
    # 改为计算收益率的高阶自相关特征
    if len(ret) >= 120:
        # 5分钟滞后自相关（捕捉短期动量持续性）
        lag5_corr = np.corrcoef(ret[:-5], ret[5:])[0, 1]
        register_factor('ret_autocorr_5', 'corr(ret[t], ret[t-5])', 
                        '5分钟滞后自相关，高值表示短期动量持续性强', source='基础因子')
        if not np.isnan(lag5_corr):
            result['ret_autocorr_5'] = float(lag5_corr)
        
        # 30分钟滞后自相关（捕捉中期趋势）
        lag30_corr = np.corrcoef(ret[:-30], ret[30:])[0, 1]
        register_factor('ret_autocorr_30', 'corr(ret[t], ret[t-30])', 
                        '30分钟滞后自相关，高值表示中期趋势持续', source='基础因子')
        if not np.isnan(lag30_corr):
            result['ret_autocorr_30'] = float(lag30_corr)
    
    # ========== 【22】价量配合因子 ==========
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        
        # 价量同向性：价格上涨时成交量增加的程度
        ret_diff = np.diff(ret)
        vol_diff = np.diff(vol_aligned)
        
        if len(ret_diff) > 30:
            # 价量同向性
            same_dir = ((ret_diff > 0) & (vol_diff > 0)) | ((ret_diff < 0) & (vol_diff < 0))
            register_factor('pv_sync', 'count(same_direction) / N', 
                            '价量同向性，高值表示价量配合好', source='东吴证券2022.08')
            result['pv_sync'] = float(same_dir.sum() / len(ret_diff))
            
            # 价量相关性
            valid = ~(np.isnan(ret_diff) | np.isnan(vol_diff))
            if valid.sum() > 30:
                pv_corr = np.corrcoef(ret_diff[valid], vol_diff[valid])[0, 1]
                register_factor('pv_corr', 'corr(ret_diff, vol_diff)', 
                                '价量变化相关性', source='东吴证券2022.08')
                if not np.isnan(pv_corr):
                    result['pv_corr'] = float(pv_corr)
    
    # ========== 【23】大小单资金流因子 ==========
    if 'taker_imb' in group.columns:
        taker_imb = group['taker_imb'].dropna().values
        
        if len(taker_imb) >= n:
            taker_aligned = taker_imb[:n]
            
            # 主动资金流与收益率的关系
            valid = ~(np.isnan(ret) | np.isnan(taker_aligned))
            if valid.sum() > 30:
                # 大单残差：用收益率解释资金流后的残差
                taker_valid = taker_aligned[valid]
                ret_valid = ret[valid]
                
                # 简单回归
                if ret_valid.std() > 0:
                    beta = np.cov(taker_valid, ret_valid)[0, 1] / (ret_valid.var() + 1e-8)
                    residual = taker_valid - beta * ret_valid
                    
                    register_factor('taker_residual', 'taker - beta * ret', 
                                    '大单残差因子，剥离收益后的资金流信息', source='开源证券2022.12')
                    result['taker_residual'] = float(residual.mean())
                    
                    # 残差的稳定性
                    register_factor('taker_residual_std', 'std(taker_residual)', 
                                    '大单残差波动', source='开源证券2022.12')
                    result['taker_residual_std'] = float(residual.std())
    
    # ========== 【24】日内收益率偏度因子 ==========
    if len(ret) >= 30:
        # SKEW因子
        skew_val = stats.skew(ret)
        register_factor('intraday_skew', 'skew(ret)', 
                        '日内收益率偏度，正偏表示右尾肥', source='开源证券2022.12')
        result['intraday_skew'] = float(skew_val)
    
    # ========== 【25】亚洲早盘因子 ==========
    # UTC 00:00-04:00 = 北京 08:00-12:00 = 亚洲股市开盘时段
    # 这是亚洲散户最活跃的时段，与A股开盘时间高度重叠
    if hours is not None and len(hours) >= n:
        hours_aligned = hours[:n]
        
        # 亚洲早盘 (UTC 00:00-04:00) = 北京08:00-12:00
        asia_morning_mask = (hours_aligned >= 0) & (hours_aligned < 4)
        # 亚洲午盘 (UTC 04:00-08:00) = 北京12:00-16:00
        asia_afternoon_mask = (hours_aligned >= 4) & (hours_aligned < 8)
        
        if asia_morning_mask.sum() > 10 and asia_afternoon_mask.sum() > 10:
            asia_morning_ret = ret[asia_morning_mask].sum()
            asia_afternoon_ret = ret[asia_afternoon_mask].sum()
            
            register_factor('asia_morning_ret', 'sum(ret[UTC 0-4])', 
                            '亚洲早盘收益(UTC 0-4=北京08-12时)，与A股早盘重叠', source='基础因子')
            result['asia_morning_ret'] = float(asia_morning_ret)
            
            register_factor('asia_afternoon_ret', 'sum(ret[UTC 4-8])', 
                            '亚洲午盘收益(UTC 4-8=北京12-16时)，与A股午盘重叠', source='基础因子')
            result['asia_afternoon_ret'] = float(asia_afternoon_ret)
            
            # 亚洲早午盘动量差
            register_factor('asia_am_pm_diff', 'asia_morning - asia_afternoon', 
                            '亚洲早午盘收益差，正值表示早盘强于午盘', source='基础因子')
            result['asia_am_pm_diff'] = float(asia_morning_ret - asia_afternoon_ret)
    
    # ========== 【26】东方证券-价格冲击不对称因子 (gammabias) ==========
    # 来源: 东方证券《因子选股系列之十六》2016.11 - 非对称价格冲击
    # 原理: 衡量等量买卖单对价格的不同冲击程度
    # 公式: return_i = gamma^up · I_i · MF_i + gamma^down · (1-I_i) · MF_i
    # gammabias = (gamma^up - gamma^down) / std_error
    # 加密货币适配: 使用taker_buy_volume构建资金流指标
    if 'taker_imb' in group.columns and len(ret) >= 60:
        taker_imb = group['taker_imb'].dropna().values
        
        if len(taker_imb) >= n and volumes is not None and len(volumes) >= n:
            # 使用5分钟聚合来计算（减少噪声）
            bar_size = 5
            n_bars = n // bar_size
            
            if n_bars >= 12:  # 至少12个5分钟bar（1小时数据）
                # 聚合为5分钟bar
                ret_5m = np.array([ret[i*bar_size:(i+1)*bar_size].sum() for i in range(n_bars)])
                taker_5m = np.array([taker_imb[i*bar_size:(i+1)*bar_size].mean() for i in range(n_bars)])
                vol_5m = np.array([volumes[i*bar_size:(i+1)*bar_size].sum() for i in range(n_bars)])
                
                # 计算资金流指标 MF = taker_imbalance (已经是标准化的净主动买入比例)
                mf = taker_5m
                
                # 分离买入和卖出时段
                buy_mask = mf > 0  # 主动买入为主的时段
                sell_mask = mf < 0  # 主动卖出为主的时段
                
                if buy_mask.sum() >= 5 and sell_mask.sum() >= 5:
                    # 计算买入时段的价格冲击 (gamma_up)
                    # ret = gamma * |mf| + epsilon
                    mf_buy = np.abs(mf[buy_mask])
                    ret_buy = ret_5m[buy_mask]
                    
                    if mf_buy.std() > 0:
                        # 简单线性回归计算gamma
                        gamma_up = np.cov(ret_buy, mf_buy)[0, 1] / (mf_buy.var() + 1e-10)
                        
                        # 计算卖出时段的价格冲击 (gamma_down)
                        mf_sell = np.abs(mf[sell_mask])
                        ret_sell = ret_5m[sell_mask]
                        
                        if mf_sell.std() > 0:
                            gamma_down = np.cov(ret_sell, mf_sell)[0, 1] / (mf_sell.var() + 1e-10)
                            
                            # gammabias: 正值表示"易涨难跌"，负值表示"易跌难涨"
                            # 研报发现负gammabias有正向预测能力（反转效应）
                            gamma_diff = gamma_up - gamma_down
                            
                            register_factor('gamma_up', 'cov(ret_buy, |mf|) / var(|mf|)', 
                                            '买入时段价格冲击系数，衡量买单推升价格的能力', 
                                            source='东方证券2016.11')
                            result['gamma_up'] = float(gamma_up)
                            
                            register_factor('gamma_down', 'cov(ret_sell, |mf|) / var(|mf|)', 
                                            '卖出时段价格冲击系数，衡量卖单压低价格的能力', 
                                            source='东方证券2016.11')
                            result['gamma_down'] = float(gamma_down)
                            
                            register_factor('gammabias', 'gamma_up - gamma_down', 
                                            '价格冲击不对称性，负值表示易跌难涨，有反转效应', 
                                            source='东方证券2016.11')
                            result['gammabias'] = float(gamma_diff)
                            
                            # 标准化版本（用于截面比较）
                            # 计算残差标准误
                            resid_buy = ret_buy - gamma_up * mf_buy
                            resid_sell = ret_sell - gamma_down * mf_sell
                            se_up = resid_buy.std() / np.sqrt(len(resid_buy))
                            se_down = resid_sell.std() / np.sqrt(len(resid_sell))
                            se_total = np.sqrt(se_up**2 + se_down**2)
                            
                            if se_total > 0:
                                gammabias_t = gamma_diff / se_total
                                register_factor('gammabias_t', '(gamma_up - gamma_down) / SE', 
                                                '标准化价格冲击不对称性（t统计量）', 
                                                source='东方证券2016.11')
                                result['gammabias_t'] = float(gammabias_t)
    
    # ========== 【27】东方证券-价格冲击弹性因子 (Lambda) ==========
    # 来源: 东方证券《因子选股系列之十四》2016.11 - 流动性度量
    # 原理: r_n = λ · S_n + μ_n, λ衡量单位订单流的价格冲击
    # Lambda越高说明流动性越差，价格冲击越大
    if 'taker_imb' in group.columns and len(ret) >= 60:
        taker_imb = group['taker_imb'].dropna().values
        
        if len(taker_imb) >= n and volumes is not None and len(volumes) >= n:
            vol_aligned = volumes[:n]
            taker_aligned = taker_imb[:n]
            
            # 构建签名订单流 S = taker_imbalance * volume (方向 × 量)
            signed_flow = taker_aligned * vol_aligned
            
            valid = ~(np.isnan(ret) | np.isnan(signed_flow) | (signed_flow == 0))
            
            if valid.sum() >= 30:
                ret_valid = ret[valid]
                flow_valid = signed_flow[valid]
                
                # 标准化订单流（避免数量级问题）
                flow_std = flow_valid.std()
                if flow_std > 0:
                    flow_normalized = flow_valid / flow_std
                    
                    # 回归: ret = lambda * signed_flow + epsilon
                    lambda_coef = np.cov(ret_valid, flow_normalized)[0, 1] / (flow_normalized.var() + 1e-10)
                    
                    register_factor('price_impact_lambda', 'cov(ret, signed_flow) / var(signed_flow)', 
                                    '价格冲击弹性，高值表示流动性差，单位订单流引发更大价格变化', 
                                    source='东方证券2016.11')
                    result['price_impact_lambda'] = float(lambda_coef)
                    
                    # 计算残差（未被订单流解释的收益）
                    residual = ret_valid - lambda_coef * flow_normalized
                    
                    register_factor('lambda_residual_std', 'std(ret - lambda * flow)', 
                                    '价格冲击残差波动，反映非订单流因素的价格波动', 
                                    source='东方证券2016.11')
                    result['lambda_residual_std'] = float(residual.std())
                    
                    # Lambda的稳定性（滚动计算的变异系数）
                    window = 60
                    if valid.sum() >= window * 2:
                        lambda_rolling = []
                        for i in range(0, valid.sum() - window, window // 2):
                            ret_w = ret_valid[i:i+window]
                            flow_w = flow_normalized[i:i+window]
                            if flow_w.var() > 0:
                                lambda_w = np.cov(ret_w, flow_w)[0, 1] / (flow_w.var() + 1e-10)
                                lambda_rolling.append(lambda_w)
                        
                        if len(lambda_rolling) >= 3:
                            lambda_arr = np.array(lambda_rolling)
                            lambda_cv = lambda_arr.std() / (np.abs(lambda_arr.mean()) + 1e-10)
                            register_factor('lambda_stability', '-std(lambda_rolling) / |mean(lambda_rolling)|', 
                                            'Lambda稳定性，高值表示价格冲击更稳定可预测', 
                                            source='东方证券2016.11')
                            result['lambda_stability'] = float(-lambda_cv)
    
    # ========== 【28】Amihud非流动性因子 (ILLIQ) ==========
    # 来源: 东方证券《因子选股系列之十四》2016.11 / Amihud (2002)
    # 原理: ILLIQ = Average(|r_t| / Volume_t)
    # 衡量单位成交量引发的价格变化，是经典的低频流动性代理
    if volumes is not None and len(volumes) >= n and len(ret) >= 30:
        vol_aligned = volumes[:n]
        
        # 过滤零成交量
        valid = (vol_aligned > 0) & (~np.isnan(ret))
        
        if valid.sum() >= 30:
            ret_valid = np.abs(ret[valid])
            vol_valid = vol_aligned[valid]
            
            # Amihud ILLIQ: 平均(|收益率| / 成交量)
            illiq_values = ret_valid / vol_valid
            
            # 去除极端值（可能是数据问题）
            illiq_clipped = np.clip(illiq_values, 0, np.percentile(illiq_values, 99))
            
            amihud_illiq = illiq_clipped.mean()
            
            register_factor('amihud_illiq', 'mean(|ret| / volume)', 
                            'Amihud非流动性，高值表示流动性差，价格冲击大', 
                            source='东方证券2016.11')
            result['amihud_illiq'] = float(amihud_illiq)
            
            # 对数版本（分布更正态）
            log_illiq = np.log(illiq_clipped + 1e-15).mean()
            register_factor('amihud_illiq_log', 'mean(log(|ret| / volume))', 
                            'Amihud非流动性(对数)，分布更正态化', 
                            source='东方证券2016.11')
            result['amihud_illiq_log'] = float(log_illiq)
            
            # ILLIQ的时变性（日内流动性变化）
            illiq_std = illiq_clipped.std()
            if amihud_illiq > 0:
                illiq_cv = illiq_std / amihud_illiq
                register_factor('illiq_variability', 'std(illiq) / mean(illiq)', 
                                '非流动性变异系数，高值表示日内流动性波动大', 
                                source='东方证券2016.11')
                result['illiq_variability'] = float(illiq_cv)
            
            # 分时段ILLIQ（检测流动性时段效应）
            if hours is not None and len(hours) >= n:
                hours_valid = hours[:n][valid]
                
                # 亚洲时段 vs 美国时段的流动性差异
                # 亚洲时段: UTC 0-8 = 北京08-16时
                # 美洲时段: UTC 13-22 = 纽约08-17时 EST
                asia_mask = (hours_valid >= 0) & (hours_valid < 8)
                us_mask = (hours_valid >= 13) & (hours_valid < 22)
                
                if asia_mask.sum() >= 10 and us_mask.sum() >= 10:
                    illiq_asia = illiq_clipped[asia_mask].mean()
                    illiq_us = illiq_clipped[us_mask].mean()
                    
                    register_factor('illiq_asia_us_ratio', 'illiq_asia / illiq_us', 
                                    '亚洲/美国时段非流动性比率(UTC 0-8 vs 13-22)，>1表示亚洲时段流动性更差', 
                                    source='东方证券2016.11')
                    if illiq_us > 0:
                        result['illiq_asia_us_ratio'] = float(illiq_asia / illiq_us)
    
    # ========== 【29】成交量加权价格冲击因子 ==========
    # 综合Lambda和成交量分布
    if 'taker_imb' in group.columns and volumes is not None and len(ret) >= 60:
        taker_imb = group['taker_imb'].dropna().values
        
        if len(taker_imb) >= n and len(volumes) >= n:
            vol_aligned = volumes[:n]
            taker_aligned = taker_imb[:n]
            
            # 高成交量时的价格冲击 vs 低成交量时的价格冲击
            vol_median = np.median(vol_aligned)
            high_vol_mask = vol_aligned > vol_median
            low_vol_mask = ~high_vol_mask
            
            if high_vol_mask.sum() >= 20 and low_vol_mask.sum() >= 20:
                # 高成交量时段的价格敏感度
                ret_high = ret[high_vol_mask]
                taker_high = taker_aligned[high_vol_mask]
                
                ret_low = ret[low_vol_mask]
                taker_low = taker_aligned[low_vol_mask]
                
                # 计算高/低成交量时段的价格敏感度
                if taker_high.std() > 0 and taker_low.std() > 0:
                    sens_high = np.cov(ret_high, taker_high)[0, 1] / (taker_high.var() + 1e-10)
                    sens_low = np.cov(ret_low, taker_low)[0, 1] / (taker_low.var() + 1e-10)
                    
                    register_factor('vol_conditional_impact', 'impact_high_vol / impact_low_vol',
                                    '成交量条件价格冲击比，>1表示高成交量时价格更敏感',
                                    source='东方证券2016.11')
                    if sens_low != 0:
                        result['vol_conditional_impact'] = float(sens_high / (sens_low + 1e-10))

    # ========== 【30】东方证券-日内残差高阶矩因子 (iVol, iSkew, iKurt) ==========
    # 来源: 东方证券《日内残差高阶矩与股票收益》2016.08
    # 原理: 使用市场收益回归残差计算高阶矩，剥离市场风险后的特质波动
    # 公式: r_i = α + β × r_MKT + ε_i
    #       iDVol = sqrt(sum(ε²))
    #       iDSkew = (√N × sum(ε³)) / iDVol^1.5
    #       iDKurt = (N × sum(ε⁴)) / iDVol²
    # 加密货币适配: 使用BTC作为市场因子
    if len(ret) >= 60:
        residual = None
        use_market_model = False

        # 完整版：使用BTC收益率进行市场回归
        if btc_ret is not None and len(btc_ret) >= len(ret):
            btc_aligned = btc_ret[:len(ret)]
            # 检查有效数据点
            valid_mask = ~(np.isnan(ret) | np.isnan(btc_aligned))
            valid_count = valid_mask.sum()

            if valid_count >= 30:
                ret_valid = ret[valid_mask]
                btc_valid = btc_aligned[valid_mask]

                # 市场模型回归: r_i = α + β × r_BTC + ε
                # 使用最小二乘法: β = cov(r, r_BTC) / var(r_BTC)
                btc_var = btc_valid.var()
                if btc_var > 1e-12:
                    beta = np.cov(ret_valid, btc_valid)[0, 1] / btc_var
                    alpha = ret_valid.mean() - beta * btc_valid.mean()

                    # 计算残差
                    residual_valid = ret_valid - alpha - beta * btc_valid

                    # 存储beta和alpha供分析
                    register_factor('market_beta', 'cov(ret, r_BTC) / var(r_BTC)',
                                    '市场Beta，对BTC的敏感度', source='东方证券2016.08')
                    result['market_beta'] = float(beta)

                    register_factor('market_alpha', 'mean(ret) - beta * mean(r_BTC)',
                                    '市场Alpha，超额收益', source='东方证券2016.08')
                    result['market_alpha'] = float(alpha)

                    # R² = 1 - var(residual) / var(ret)
                    r_squared = 1 - residual_valid.var() / (ret_valid.var() + 1e-12)
                    register_factor('market_r2', '1 - var(ε) / var(ret)',
                                    '市场模型R²，市场因子解释力', source='东方证券2016.08')
                    result['market_r2'] = float(r_squared)

                    residual = residual_valid
                    use_market_model = True

        # 简化版（回退）：使用去均值收益作为残差
        if residual is None:
            residual = ret - ret.mean()

        # 日内特质波动率 iVol = sqrt(sum(ε²))
        i_vol = np.sqrt((residual ** 2).sum())
        factor_suffix = '' if use_market_model else '_simple'
        factor_desc = '(市场中性)' if use_market_model else '(简化版)'

        register_factor(f'intraday_idio_vol{factor_suffix}',
                        'sqrt(sum(ε²))' if use_market_model else 'sqrt(sum((ret-mean)²))',
                        f'日内特质波动率{factor_desc}，剥离市场风险后的特质波动',
                        source='东方证券2016.08')
        result[f'intraday_idio_vol{factor_suffix}'] = float(i_vol)

        # 日内特质偏度 iSkew = (√N × sum(ε³)) / iVol^1.5
        if i_vol > 0:
            n_obs = len(residual)
            i_skew = (np.sqrt(n_obs) * (residual ** 3).sum()) / (i_vol ** 1.5)
            register_factor(f'intraday_idio_skew{factor_suffix}',
                            '(√N × sum(ε³)) / iVol^1.5',
                            f'日内特质偏度{factor_desc}，负值预示下跌风险，IC=-0.076, ICIR=-1.37',
                            source='东方证券2016.08')
            result[f'intraday_idio_skew{factor_suffix}'] = float(i_skew)

            # 日内特质峰度 iKurt = (N × sum(ε⁴)) / iVol²
            i_kurt = (n_obs * (residual ** 4).sum()) / (i_vol ** 2)
            register_factor(f'intraday_idio_kurt{factor_suffix}',
                            '(N × sum(ε⁴)) / iVol²',
                            f'日内特质峰度{factor_desc}，高值表示尾部风险大',
                            source='东方证券2016.08')
            result[f'intraday_idio_kurt{factor_suffix}'] = float(i_kurt)

            # 额外：特质偏度与特质波动的比率（研报提到的复合因子）
            if use_market_model:
                skew_vol_ratio = i_skew / (i_vol + 1e-10)
                register_factor('idio_skew_vol_ratio', 'iSkew / iVol',
                                '特质偏度/波动率比，综合风险指标', source='东方证券2016.08')
                result['idio_skew_vol_ratio'] = float(skew_vol_ratio)

    # ========== 【31】方正证券-APM因子 (Afternoon-Minus-Morning) ==========
    # 来源: 方正证券《凤鸣朝阳》2016.10
    # 原理: 上午和下午的异常收益差异反映知情交易者活动
    # 加密货币适配: 使用亚洲早盘(UTC 0-4=北京8-12) vs 亚洲午盘(UTC 4-8=北京12-16)
    # 或使用 亚洲时段(UTC 0-8) vs 美洲时段(UTC 13-22)
    if hours is not None and len(hours) >= n:
        hours_aligned = hours[:n]

        # 方案1: 亚洲早午盘差异 (与A股市场对照)
        # UTC 0-4 = 北京08-12时 = A股早盘
        # UTC 4-8 = 北京12-16时 = A股午盘
        asia_am_mask = (hours_aligned >= 0) & (hours_aligned < 4)
        asia_pm_mask = (hours_aligned >= 4) & (hours_aligned < 8)

        if asia_am_mask.sum() >= 30 and asia_pm_mask.sum() >= 30:
            ret_asia_am = ret[asia_am_mask]
            ret_asia_pm = ret[asia_pm_mask]

            # APM = 早盘异常收益 - 午盘异常收益
            # 使用当日去均值收益作为异常收益的简化
            ret_mean = ret.mean()
            apm_asia = (ret_asia_am.sum() - ret_mean * asia_am_mask.sum()) - \
                       (ret_asia_pm.sum() - ret_mean * asia_pm_mask.sum())

            register_factor('apm_asia', '(ret_am - E[ret]*N_am) - (ret_pm - E[ret]*N_pm)',
                            'APM因子(亚洲时段)，早午盘异常收益差，正值表示早盘强势',
                            source='方正证券2016.10')
            result['apm_asia'] = float(apm_asia)

            # 标准化APM (用于截面比较)
            ret_std = ret.std()
            if ret_std > 0:
                apm_asia_std = apm_asia / (ret_std * np.sqrt(asia_am_mask.sum() + asia_pm_mask.sum()))
                register_factor('apm_asia_std', 'apm_asia / (std * sqrt(N))',
                                '标准化APM因子(亚洲时段)', source='方正证券2016.10')
                result['apm_asia_std'] = float(apm_asia_std)

        # 方案2: 亚洲vs美洲时段差异 (加密市场特有)
        # UTC 0-8 = 亚洲时段
        # UTC 13-22 = 美洲时段
        asia_mask = (hours_aligned >= 0) & (hours_aligned < 8)
        us_mask = (hours_aligned >= 13) & (hours_aligned < 22)

        if asia_mask.sum() >= 60 and us_mask.sum() >= 60:
            ret_asia = ret[asia_mask]
            ret_us = ret[us_mask]

            # 亚洲-美洲收益差
            apm_global = ret_asia.sum() - ret_us.sum()
            register_factor('apm_global', 'ret_asia - ret_us',
                            'APM因子(全球时段)，亚洲vs美洲收益差', source='方正证券2016.10')
            result['apm_global'] = float(apm_global)

            # 波动率调整版本
            asia_vol = ret_asia.std()
            us_vol = ret_us.std()
            if asia_vol > 0 and us_vol > 0:
                apm_vol_adj = (ret_asia.mean() / asia_vol) - (ret_us.mean() / us_vol)
                register_factor('apm_vol_adjusted', 'mean(ret_asia)/std - mean(ret_us)/std',
                                'APM因子(波动率调整)，风险调整后的时段差异', source='方正证券2016.10')
                result['apm_vol_adjusted'] = float(apm_vol_adj)

    # ========== 【32】成交量分布均匀度因子 (Volume Uniformity) ==========
    # 来源: 多篇研报提及的成交量时间分布特征
    # 原理: 成交量分布越均匀，表示交易活动越稳定，信息释放越平滑
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        total_vol = vol_aligned.sum()

        if total_vol > 0:
            # 方法1: 使用熵衡量成交量分布均匀度
            # 将一天分成24小时，计算每小时成交量占比的熵
            if hours is not None and len(hours) >= n:
                hours_aligned = hours[:n]
                hourly_vol = np.array([vol_aligned[hours_aligned == h].sum() for h in range(24)])
                hourly_vol = hourly_vol[hourly_vol > 0]  # 去除零值

                if len(hourly_vol) > 1:
                    # 计算成交量分布熵
                    vol_prob = hourly_vol / hourly_vol.sum()
                    vol_entropy = -np.sum(vol_prob * np.log(vol_prob + 1e-10))
                    max_entropy = np.log(len(hourly_vol))

                    register_factor('vol_dist_entropy', '-sum(p_vol * log(p_vol))',
                                    '成交量分布熵，高值表示成交量时间分布均匀', source='基础因子')
                    result['vol_dist_entropy'] = float(vol_entropy)

                    # 归一化熵 (0-1之间)
                    if max_entropy > 0:
                        vol_uniformity = vol_entropy / max_entropy
                        register_factor('vol_uniformity', 'entropy / max_entropy',
                                        '成交量均匀度(0-1)，越接近1表示分布越均匀', source='基础因子')
                        result['vol_uniformity'] = float(vol_uniformity)

            # 方法2: 使用基尼系数衡量成交量集中度
            # 将分钟成交量排序，计算洛伦兹曲线下面积
            vol_sorted = np.sort(vol_aligned)
            n_vol = len(vol_sorted)
            cum_vol = np.cumsum(vol_sorted)

            # 基尼系数 = 1 - 2 * (洛伦兹曲线下面积)
            # 简化计算: G = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
            idx = np.arange(1, n_vol + 1)
            gini = (2 * np.sum(idx * vol_sorted) - (n_vol + 1) * total_vol) / (n_vol * total_vol)

            register_factor('vol_gini', '基尼系数计算公式',
                            '成交量基尼系数，高值表示成交量集中在少数时段', source='基础因子')
            result['vol_gini'] = float(gini)

            # 反转为均匀度因子 (基尼系数越低越均匀)
            register_factor('vol_equality', '1 - gini',
                            '成交量平等度，高值表示成交量分布均匀', source='基础因子')
            result['vol_equality'] = float(1 - gini)

    # ========== 【33】CPV因子 (Cumulative Price-Volume Correlation) ==========
    # 来源: 多篇高频因子研报
    # 原理: 累积收益与累积成交量的相关性，反映量价配合程度
    if volumes is not None and len(volumes) >= n and len(ret) >= 30:
        vol_aligned = volumes[:n]

        # 累积收益和累积成交量
        cum_ret = np.cumsum(ret)
        cum_vol = np.cumsum(vol_aligned)

        # 标准化
        if cum_vol.std() > 0 and cum_ret.std() > 0:
            # CPV: 累积收益与累积成交量的相关系数
            cpv = np.corrcoef(cum_ret, cum_vol)[0, 1]

            register_factor('cpv', 'corr(cumsum(ret), cumsum(vol))',
                            'CPV因子，累积收益与累积成交量相关性，正值表示放量上涨',
                            source='基础因子')
            if not np.isnan(cpv):
                result['cpv'] = float(cpv)

            # 分段CPV (前半段vs后半段)
            mid = n // 2
            if mid >= 30:
                cpv_first = np.corrcoef(cum_ret[:mid], cum_vol[:mid])[0, 1]
                cpv_second = np.corrcoef(cum_ret[mid:] - cum_ret[mid-1],
                                          cum_vol[mid:] - cum_vol[mid-1])[0, 1]

                if not np.isnan(cpv_first):
                    register_factor('cpv_first_half', 'corr(cumret, cumvol)[0:N/2]',
                                    'CPV因子(前半段)', source='基础因子')
                    result['cpv_first_half'] = float(cpv_first)

                if not np.isnan(cpv_second):
                    register_factor('cpv_second_half', 'corr(cumret, cumvol)[N/2:N]',
                                    'CPV因子(后半段)', source='基础因子')
                    result['cpv_second_half'] = float(cpv_second)

                # CPV变化 (后半段-前半段)
                if not np.isnan(cpv_first) and not np.isnan(cpv_second):
                    register_factor('cpv_change', 'cpv_second - cpv_first',
                                    'CPV变化，正值表示量价配合后半段改善', source='基础因子')
                    result['cpv_change'] = float(cpv_second - cpv_first)

        # RPV因子: 收益率与成交量的滚动相关性
        window = 60  # 1小时窗口
        if n >= window * 2:
            rpv_list = []
            for i in range(0, n - window + 1, window // 2):
                ret_w = ret[i:i+window]
                vol_w = vol_aligned[i:i+window]
                if ret_w.std() > 0 and vol_w.std() > 0:
                    rpv_w = np.corrcoef(ret_w, vol_w)[0, 1]
                    if not np.isnan(rpv_w):
                        rpv_list.append(rpv_w)

            if len(rpv_list) >= 3:
                rpv_arr = np.array(rpv_list)

                register_factor('rpv_mean', 'mean(rolling_corr(ret, vol))',
                                'RPV因子均值，滚动量价相关性平均', source='基础因子')
                result['rpv_mean'] = float(rpv_arr.mean())

                register_factor('rpv_std', 'std(rolling_corr(ret, vol))',
                                'RPV因子波动，量价关系稳定性', source='基础因子')
                result['rpv_std'] = float(rpv_arr.std())

                # RPV趋势 (量价关系是否在改善)
                if len(rpv_arr) >= 4:
                    rpv_trend = rpv_arr[-2:].mean() - rpv_arr[:2].mean()
                    register_factor('rpv_trend', 'rpv_late - rpv_early',
                                    'RPV趋势，正值表示量价配合在改善', source='基础因子')
                    result['rpv_trend'] = float(rpv_trend)

    return result


def compute_intraday_factors_for_day(group: pd.DataFrame, symbol: str, date, price_col: str,
                                      btc_ret: Optional[np.ndarray] = None) -> Optional[Dict]:
    """计算单日所有日内因子

    Args:
        group: 单日分钟级数据
        symbol: 币种符号
        date: 日期
        price_col: 价格列名
        btc_ret: BTC分钟收益率数组（用于市场回归计算残差）
    """
    # 先计算基础因子
    base_result = compute_base_factors(group, symbol, date, price_col)
    if base_result is None:
        return None

    # 再计算增强因子（传入BTC收益率用于市场回归）
    enhanced_result = compute_enhanced_factors(group, symbol, date, price_col, btc_ret)

    # 合并结果
    base_result.update(enhanced_result)

    return base_result


def load_btc_minute_returns(data_dir: str) -> Optional[pd.DataFrame]:
    """
    加载BTC分钟收益率数据，用于计算市场中性残差
    返回: 以open_time为索引的DataFrame，包含btc_ret列
    """
    btc_path = Path(data_dir) / "BTCUSDT.parquet"
    if not btc_path.exists():
        print("Warning: BTCUSDT.parquet not found, will use simplified residual calculation")
        return None

    print("Loading BTC minute data for market regression...")
    btc_df = pd.read_parquet(btc_path)
    btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])

    price_col = 'mark_close' if 'mark_close' in btc_df.columns else 'close'
    btc_df['btc_ret'] = btc_df[price_col].pct_change()

    # 只保留需要的列，以open_time为索引（用于快速合并）
    btc_returns = btc_df[['open_time', 'btc_ret']].set_index('open_time')

    print(f"  Loaded {len(btc_returns):,} BTC minute returns")
    del btc_df
    gc.collect()

    return btc_returns


def extract_intraday_factors(symbol_files: Dict[str, Path], data_dir: str) -> pd.DataFrame:
    """从分钟数据提取日内因子"""
    print("\nExtracting intraday factors...")

    # 先加载BTC分钟收益率（用于市场回归）- 返回DataFrame而非字典
    btc_df = load_btc_minute_returns(data_dir)

    factors_list = []
    total_symbols = len(symbol_files)
    processed = 0

    for symbol, file_path in symbol_files.items():
        processed += 1
        if processed % 50 == 0:
            print(f"  Processing symbol {processed}/{total_symbols}: {symbol}")

        try:
            df = pd.read_parquet(file_path)

            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            df['open_time'] = pd.to_datetime(df['open_time'])
            df['date'] = df['open_time'].dt.date
            df['hour'] = df['open_time'].dt.hour

            price_col = 'mark_close' if 'mark_close' in df.columns else 'close'

            # 计算收益率
            df['ret'] = df[price_col].pct_change()
            df['avg_trade_size'] = df['volume'] / (df['trades'] + 1e-8)

            # Taker imbalance
            if 'taker_buy_volume' in df.columns:
                df['taker_imb'] = (2 * df['taker_buy_volume'] - df['volume']) / (df['volume'] + 1e-8)

            # 合并BTC收益率（使用pandas索引合并，比字典查找快100倍以上）
            if btc_df is not None and symbol != 'BTCUSDT':
                df = df.set_index('open_time').join(btc_df, how='left').reset_index()
            else:
                df['btc_ret'] = np.nan

            # 按日期分组计算
            for date, group in df.groupby('date'):
                if len(group) < 60:  # 至少60分钟数据
                    continue

                # BTC收益率已经在DataFrame中，直接提取
                btc_ret_day = group['btc_ret'].values if 'btc_ret' in group.columns else None

                row = compute_intraday_factors_for_day(group, symbol, date, price_col, btc_ret_day)
                if row:
                    factors_list.append(row)
            
            del df
            gc.collect()
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue
    
    if not factors_list:
        return pd.DataFrame()
    
    result = pd.DataFrame(factors_list)
    factor_cols = [c for c in result.columns if c not in ['symbol', 'date']]
    print(f"Extracted {len(result):,} rows, {len(factor_cols)} intraday factors")
    
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    """
    因子挖掘主函数

    职责：从分钟数据计算日内因子，保存到 factors/intraday_factors.csv
    因子筛选由 factor_screening.py 负责
    """
    parser = argparse.ArgumentParser(description='Crypto Intraday Factor Mining')
    parser.add_argument('--data_dir', type=str, default='./futures_data_1m')

    args = parser.parse_args()

    print("=" * 80)
    print("CRYPTO INTRADAY FACTOR MINING")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Pool size: {len(QUALIFIED_POOL)} symbols")
    print("=" * 80)

    # 1. 加载分钟数据
    symbol_files, daily_df = load_minute_data_and_resample(args.data_dir)

    # 2. 提取日内因子
    intraday_df = extract_intraday_factors(symbol_files, args.data_dir)

    if intraday_df.empty:
        print("No intraday factors extracted!")
        return

    # 3. 保存因子值
    factors_dir = Path('./factors')
    factors_dir.mkdir(exist_ok=True)
    intraday_df.to_csv(factors_dir / 'intraday_factors.csv', index=False)

    factor_cols = [c for c in intraday_df.columns if c not in ['symbol', 'date']]
    print(f"\nSaved: ./factors/intraday_factors.csv")
    print(f"  - Rows: {len(intraday_df):,}")
    print(f"  - Factors: {len(factor_cols)}")
    print(f"  - Symbols: {intraday_df['symbol'].nunique()}")
    print(f"  - Date range: {intraday_df['date'].min()} ~ {intraday_df['date'].max()}")

    print("\n" + "=" * 80)
    print("DONE! Run factor_screening.py to analyze and filter factors.")
    print("=" * 80)


if __name__ == "__main__":
    main()