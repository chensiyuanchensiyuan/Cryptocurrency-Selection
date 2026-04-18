"""
加密货币因子筛选脚本 - 等权版本

方法论：
- 等权分层：每个标的同等权重，统计上最公平
- 动态Winsorize：每日横截面收益按1%-99%分位数截断

筛选标准：
- |IC| >= 0.02
- 多空策略Sharpe >= 1.0

分层方法：
- 按因子值排序分5层（quintile）
- 层内等权平均计算收益
- 做多top 20%（第5层），做空bottom 20%（第1层）
- 杠杆：1倍（50%资金做多 + 50%资金做空）

使用方法:
    python factor_screening.py
    python factor_screening.py --daily_factors ./factors/daily_factors.csv --intraday_factors ./factors/intraday_factors.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============== 配置 ==============
FILTER_THRESHOLDS = {
    'min_ic_abs': 0.02,       # IC绝对值最小阈值
    'min_sharpe': 1.0,        # 多空策略夏普最小阈值
    'n_quantiles': 5,         # 分层数量
    'long_quantile': 5,       # 做多的层（最高层）
    'short_quantile': 1,      # 做空的层（最低层）
}

WINSORIZE_LOWER = 1   # 下分位数 1%
WINSORIZE_UPPER = 99  # 上分位数 99%

DEDUP_CORR_THRESHOLD = 0.99  # 因子去重相关性阈值


# ============================================================
# 数据加载
# ============================================================

def load_factors(daily_path: str, intraday_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载因子数据"""
    print("Loading factor data...")
    
    daily_df = None
    intraday_df = None
    
    if Path(daily_path).exists():
        daily_df = pd.read_csv(daily_path)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        print(f"  Daily factors: {daily_df.shape}")
    else:
        print(f"  Warning: {daily_path} not found")
    
    if Path(intraday_path).exists():
        intraday_df = pd.read_csv(intraday_path)
        intraday_df['date'] = pd.to_datetime(intraday_df['date'])
        print(f"  Intraday factors: {intraday_df.shape}")
    else:
        print(f"  Warning: {intraday_path} not found")
    
    return daily_df, intraday_df


def load_price_data(data_dir: str) -> pd.DataFrame:
    """加载价格数据用于计算前向收益"""
    import glob
    
    print(f"Loading price data from {data_dir}...")
    
    files = glob.glob(f"{data_dir}/*.parquet")
    if not files:
        files = glob.glob(f"{data_dir}/*.csv")
    
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    dfs = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files loaded")
    
    data = pd.concat(dfs, ignore_index=True)
    
    # 处理时间列
    time_col = 'open_time' if 'open_time' in data.columns else 'date'
    data['date'] = pd.to_datetime(data[time_col])
    
    # 只保留日期部分
    if data['date'].dt.time.iloc[0] != pd.Timestamp('00:00:00').time():
        data['date'] = data['date'].dt.normalize()
    
    # 价格列
    price_col = 'mark_close' if 'mark_close' in data.columns else 'close'
    
    # 计算前向收益
    data = data.sort_values(['symbol', 'date'])
    data['fwd_ret'] = data.groupby('symbol')[price_col].shift(-1) / data[price_col] - 1
    
    print(f"  Loaded {data['symbol'].nunique()} symbols")
    print(f"  Date range: {data['date'].min().date()} ~ {data['date'].max().date()}")
    
    return data[['date', 'symbol', 'fwd_ret']].copy()


# ============================================================
# Winsorize函数
# ============================================================

def compute_daily_winsorize_thresholds(price_df: pd.DataFrame, 
                                        lower_pct: float = 1, 
                                        upper_pct: float = 99) -> pd.DataFrame:
    """
    计算每日的winsorize阈值
    
    Args:
        price_df: 包含 date, symbol, fwd_ret 的DataFrame
        lower_pct: 下分位数百分比
        upper_pct: 上分位数百分比
    
    Returns:
        DataFrame with columns: date, lower_threshold, upper_threshold, n_symbols
    """
    thresholds = []
    
    for date, group in price_df.groupby('date'):
        ret = group['fwd_ret'].dropna()
        if len(ret) < 10:
            continue
        
        lower = np.percentile(ret, lower_pct)
        upper = np.percentile(ret, upper_pct)
        
        thresholds.append({
            'date': date,
            'lower_threshold': lower,
            'upper_threshold': upper,
            'n_symbols': len(ret),
            'raw_min': ret.min(),
            'raw_max': ret.max(),
            'raw_mean': ret.mean(),
            'raw_std': ret.std(),
        })
    
    return pd.DataFrame(thresholds)


def apply_daily_winsorize(price_df: pd.DataFrame, 
                          thresholds_df: pd.DataFrame) -> pd.DataFrame:
    """
    应用每日winsorize阈值
    
    Args:
        price_df: 包含 date, symbol, fwd_ret 的DataFrame
        thresholds_df: 每日阈值DataFrame
    
    Returns:
        应用winsorize后的DataFrame
    """
    df = price_df.merge(thresholds_df[['date', 'lower_threshold', 'upper_threshold']], 
                        on='date', how='left')
    
    df['fwd_ret_win'] = df['fwd_ret'].clip(
        lower=df['lower_threshold'], 
        upper=df['upper_threshold']
    )
    
    return df[['date', 'symbol', 'fwd_ret', 'fwd_ret_win']].copy()


# ============================================================
# 因子分析函数
# ============================================================

def compute_ic(factor_values: np.ndarray, forward_returns: np.ndarray) -> float:
    """计算Spearman IC"""
    valid = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    if valid.sum() < 20:
        return np.nan
    
    ic, _ = stats.spearmanr(factor_values[valid], forward_returns[valid])
    return ic


def compute_quantile_equal_weight_returns(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    n_quantiles: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算等权分层收益
    
    方法：
    1. 按因子值排序分成n_quantiles层
    2. 每层内等权平均计算收益
    
    Returns:
        quantile_returns: shape (n_quantiles,) 每层的等权收益
        quantile_counts: shape (n_quantiles,) 每层的股票数量
    """
    valid = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    
    if valid.sum() < n_quantiles * 5:  # 每层至少5个
        return np.full(n_quantiles, np.nan), np.zeros(n_quantiles)
    
    fc = factor_values[valid]
    ret = forward_returns[valid]
    n = len(fc)
    
    # 按因子值排序
    sorted_idx = np.argsort(fc)
    quintile_size = n // n_quantiles
    
    quantile_returns = np.zeros(n_quantiles)
    quantile_counts = np.zeros(n_quantiles)
    
    for i in range(n_quantiles):
        if i < n_quantiles - 1:
            group_idx = sorted_idx[i * quintile_size : (i + 1) * quintile_size]
        else:
            group_idx = sorted_idx[i * quintile_size :]  # 最后一组包含剩余
        
        # 等权平均
        quantile_returns[i] = ret[group_idx].mean()
        quantile_counts[i] = len(group_idx)
    
    return quantile_returns, quantile_counts


def analyze_factor(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_name: str,
    test_start: str,
    n_quantiles: int = 5,
    long_q: int = 5,
    short_q: int = 1
) -> Optional[Dict]:
    """
    分析单个因子 - 等权版本

    Args:
        factor_df: 因子数据，包含 date, symbol, factor_name 列
        price_df: 价格数据，包含 date, symbol, fwd_ret_win 列（已winsorize）
        factor_name: 因子名称
        test_start: 测试开始日期
        n_quantiles: 分层数量
        long_q: 做多的层
        short_q: 做空的层

    Returns:
        分析结果字典，包含 direction 字段
        - 如果 IC > 0：direction = 1，指标为原始值
        - 如果 IC < 0：direction = -1，指标为取反后的值（IC变正，Sharpe变正）
    """
    # 合并因子和价格数据
    merged = factor_df[['date', 'symbol', factor_name]].merge(
        price_df[['date', 'symbol', 'fwd_ret_win']], on=['date', 'symbol'], how='inner'
    )

    # 筛选测试期
    merged = merged[merged['date'] >= test_start].copy()

    if len(merged) < 100:
        return None

    dates = sorted(merged['date'].unique())

    ic_list = []
    long_ret_list = []
    short_ret_list = []
    ls_ret_list = []
    valid_counts = []

    for date in dates:
        day_data = merged[merged['date'] == date]

        fc = day_data[factor_name].values
        ret = day_data['fwd_ret_win'].values

        # 计算IC
        ic = compute_ic(fc, ret)
        ic_list.append(ic)

        # 计算等权分层收益
        q_returns, q_counts = compute_quantile_equal_weight_returns(fc, ret, n_quantiles)
        valid_counts.append(q_counts.sum())

        if not np.isnan(q_returns).any():
            long_ret = q_returns[long_q - 1]   # 做多最高层 (Q5)
            short_ret = q_returns[short_q - 1] # 做空最低层 (Q1)

            long_ret_list.append(long_ret)
            short_ret_list.append(-short_ret)  # 做空收益 = -底层收益
            ls_ret_list.append(0.5 * long_ret + 0.5 * (-short_ret))  # 多空收益 (50%做多 + 50%做空)
        else:
            long_ret_list.append(np.nan)
            short_ret_list.append(np.nan)
            ls_ret_list.append(np.nan)
    
    # 转换为数组
    ic_arr = np.array(ic_list)
    long_arr = np.array(long_ret_list)
    short_arr = np.array(short_ret_list)
    ls_arr = np.array(ls_ret_list)
    
    # 去除nan
    ic_valid = ic_arr[~np.isnan(ic_arr)]
    long_valid = long_arr[~np.isnan(long_arr)]
    short_valid = short_arr[~np.isnan(short_arr)]
    ls_valid = ls_arr[~np.isnan(ls_arr)]
    
    if len(ic_valid) < 50 or len(ls_valid) < 50:
        return None
    
    # IC统计
    ic_mean = ic_valid.mean()
    ic_std = ic_valid.std()
    ic_ir = ic_mean / (ic_std + 1e-8)
    ic_positive_ratio = (ic_valid > 0).mean()
    
    # 纯多头统计
    long_mean = long_valid.mean()
    long_std = long_valid.std()
    long_sharpe = long_mean / (long_std + 1e-8) * np.sqrt(365)
    cum_long = np.cumprod(1 + np.nan_to_num(long_arr, 0))
    
    # 纯空头统计
    short_mean = short_valid.mean()
    short_std = short_valid.std()
    short_sharpe = short_mean / (short_std + 1e-8) * np.sqrt(365)
    cum_short = np.cumprod(1 + np.nan_to_num(short_arr, 0))
    
    # 多空组合统计
    ls_mean = ls_valid.mean()
    ls_std = ls_valid.std()
    ls_sharpe = ls_mean / (ls_std + 1e-8) * np.sqrt(365)
    
    # 净值和回撤
    cum_nav = np.cumprod(1 + np.nan_to_num(ls_arr, 0))
    max_dd = (cum_nav / np.maximum.accumulate(cum_nav) - 1).min()
    
    # 几何年化收益（从净值反推）
    n_years = len(ls_valid) / 365
    long_final = cum_long[-1] if len(cum_long) > 0 else 1.0
    short_final = cum_short[-1] if len(cum_short) > 0 else 1.0
    ls_final = cum_nav[-1] if len(cum_nav) > 0 else 1.0
    
    long_ann_ret = (long_final ** (1 / n_years)) - 1 if n_years > 0 else 0
    short_ann_ret = (short_final ** (1 / n_years)) - 1 if n_years > 0 else 0
    ls_ann_ret = (ls_final ** (1 / n_years)) - 1 if n_years > 0 else 0
    
    # 年度收益
    yearly = {}
    for i, date in enumerate(dates):
        if i < len(ls_arr) and not np.isnan(ls_arr[i]):
            yr = date.year
            if yr not in yearly:
                yearly[yr] = []
            yearly[yr].append(ls_arr[i])
    
    yearly_ret = {yr: np.prod(1 + np.array(rets)) - 1 for yr, rets in yearly.items() if len(rets) > 0}

    # 根据 IC 符号确定方向
    # direction = 1: IC > 0，因子值越大收益越高
    # direction = -1: IC < 0，因子值越小收益越高（使用时需取反）
    direction = 1 if ic_mean >= 0 else -1

    return {
        'name': factor_name,
        'direction': direction,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_positive_ratio': ic_positive_ratio,
        'long_sharpe': long_sharpe,
        'long_ann_ret': long_ann_ret,
        'short_sharpe': short_sharpe,
        'short_ann_ret': short_ann_ret,
        'sharpe': ls_sharpe,
        'ann_ret': ls_ann_ret,
        'max_dd': max_dd,
        'final_nav': cum_nav[-1] if len(cum_nav) > 0 else 1.0,
        'yearly_ret': yearly_ret,
        'n_days': len(ic_valid),
        'avg_valid_count': np.mean(valid_counts),
    }


def filter_qualified_factors(results: Dict, thresholds: Dict) -> List[str]:
    """筛选合格因子"""
    qualified = []
    for name, r in results.items():
        # |IC| >= 阈值 AND Sharpe >= 阈值
        if (abs(r['ic_mean']) >= thresholds['min_ic_abs'] and
            abs(r['sharpe']) >= thresholds['min_sharpe']):
            qualified.append(name)
    return qualified


def dedup_factors_by_correlation(
    qualified: List[str],
    results: Dict,
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    corr_threshold: float = 0.99
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    基于相关性去除重复因子

    Args:
        qualified: 合格因子列表
        results: 因子分析结果
        daily_df: 日频因子数据
        intraday_df: 日内因子数据
        corr_threshold: 相关性阈值，超过此值认为重复

    Returns:
        (去重后的因子列表, 被去除的因子对列表[(removed, kept, corr)])
    """
    if len(qualified) <= 1:
        return qualified, []

    # 构建因子宽表
    factor_wide = {}
    for factor_name in qualified:
        # 确定因子来源
        if daily_df is not None and factor_name in daily_df.columns:
            df = daily_df[['date', 'symbol', factor_name]].copy()
        elif intraday_df is not None and factor_name in intraday_df.columns:
            df = intraday_df[['date', 'symbol', factor_name]].copy()
        else:
            continue

        # 转换为宽表
        try:
            wide = df.pivot(index='date', columns='symbol', values=factor_name)
            # 展平为一维序列用于计算相关性
            factor_wide[factor_name] = wide.values.flatten()
        except:
            continue

    if len(factor_wide) <= 1:
        return qualified, []

    # 按 Sharpe 降序排列（保留 Sharpe 更高的）
    sorted_factors = sorted(qualified, key=lambda x: results[x]['sharpe'], reverse=True)

    # 去重
    kept = []
    removed_pairs = []

    for factor in sorted_factors:
        if factor not in factor_wide:
            kept.append(factor)
            continue

        is_duplicate = False
        for kept_factor in kept:
            if kept_factor not in factor_wide:
                continue

            # 计算相关性
            f1 = factor_wide[factor]
            f2 = factor_wide[kept_factor]

            # 对齐长度并去除 NaN
            min_len = min(len(f1), len(f2))
            f1, f2 = f1[:min_len], f2[:min_len]
            valid = ~(np.isnan(f1) | np.isnan(f2))

            if valid.sum() < 100:
                continue

            corr = np.corrcoef(f1[valid], f2[valid])[0, 1]

            if abs(corr) > corr_threshold:
                is_duplicate = True
                removed_pairs.append((factor, kept_factor, corr))
                break

        if not is_duplicate:
            kept.append(factor)

    return kept, removed_pairs


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Factor Screening - Equal Weight')
    parser.add_argument('--daily_factors', type=str, default='./factors/daily_factors.csv')
    parser.add_argument('--intraday_factors', type=str, default='./factors/intraday_factors.csv')
    parser.add_argument('--price_data', type=str, default='./futures_data',
                        help='价格数据目录')
    parser.add_argument('--test_start', type=str, default='2022-01-01')
    parser.add_argument('--output_dir', type=str, default='./screening_results')
    parser.add_argument('--winsorize_lower', type=float, default=1.0,
                        help='Winsorize下分位数 (默认1%%)')
    parser.add_argument('--winsorize_upper', type=float, default=99.0,
                        help='Winsorize上分位数 (默认99%%)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("FACTOR SCREENING - EQUAL WEIGHT")
    print("=" * 80)
    print(f"Daily factors: {args.daily_factors}")
    print(f"Intraday factors: {args.intraday_factors}")
    print(f"Price data: {args.price_data}")
    print(f"Test start: {args.test_start}")
    print(f"Thresholds: |IC| >= {FILTER_THRESHOLDS['min_ic_abs']}, Sharpe >= {FILTER_THRESHOLDS['min_sharpe']}")
    print(f"Winsorize: {args.winsorize_lower}% - {args.winsorize_upper}%")
    print(f"Method: Equal weight within quantiles")
    print("=" * 80)
    
    # 1. 加载因子数据
    daily_df, intraday_df = load_factors(args.daily_factors, args.intraday_factors)
    
    # 2. 加载价格数据
    price_df = load_price_data(args.price_data)
    
    # 3. 计算每日winsorize阈值
    print("\n" + "=" * 80)
    print("COMPUTING DAILY WINSORIZE THRESHOLDS...")
    print("=" * 80)
    
    thresholds_df = compute_daily_winsorize_thresholds(
        price_df, args.winsorize_lower, args.winsorize_upper
    )
    
    # 保存阈值CSV
    thresholds_path = output_dir / 'daily_winsorize_thresholds.csv'
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"Saved daily thresholds: {thresholds_path}")
    
    # 打印阈值统计
    print(f"\nThreshold statistics:")
    print(f"  Date range: {thresholds_df['date'].min().date()} ~ {thresholds_df['date'].max().date()}")
    print(f"  Lower threshold: {thresholds_df['lower_threshold'].mean():.2%} (avg), "
          f"[{thresholds_df['lower_threshold'].min():.2%}, {thresholds_df['lower_threshold'].max():.2%}]")
    print(f"  Upper threshold: {thresholds_df['upper_threshold'].mean():.2%} (avg), "
          f"[{thresholds_df['upper_threshold'].min():.2%}, {thresholds_df['upper_threshold'].max():.2%}]")
    
    # 4. 应用winsorize
    print("\nApplying daily winsorize...")
    price_df = apply_daily_winsorize(price_df, thresholds_df)
    
    # 统计winsorize效果
    n_clipped = (price_df['fwd_ret'] != price_df['fwd_ret_win']).sum()
    n_total = price_df['fwd_ret'].notna().sum()
    print(f"  Clipped: {n_clipped:,} / {n_total:,} ({n_clipped/n_total:.2%})")
    
    # 5. 获取因子列表
    daily_factor_names = []
    intraday_factor_names = []
    
    if daily_df is not None:
        daily_factor_names = [c for c in daily_df.columns if c not in ['date', 'symbol']]
        print(f"\nDaily factors to analyze: {len(daily_factor_names)}")
    
    if intraday_df is not None:
        intraday_factor_names = [c for c in intraday_df.columns if c not in ['date', 'symbol']]
        print(f"Intraday factors to analyze: {len(intraday_factor_names)}")
    
    # 6. 分析因子
    print("\n" + "=" * 80)
    print("ANALYZING FACTORS...")
    print("=" * 80)
    
    results = {}
    
    # 分析日频因子
    if daily_df is not None:
        print("\nAnalyzing daily factors...")
        for i, factor_name in enumerate(daily_factor_names):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(daily_factor_names)}")
            
            try:
                r = analyze_factor(
                    daily_df, price_df, factor_name, args.test_start,
                    FILTER_THRESHOLDS['n_quantiles'],
                    FILTER_THRESHOLDS['long_quantile'],
                    FILTER_THRESHOLDS['short_quantile']
                )
                if r:
                    r['type'] = 'Daily'
                    results[factor_name] = r
            except Exception as e:
                print(f"  Error analyzing {factor_name}: {e}")
    
    # 分析日内因子
    if intraday_df is not None:
        print("\nAnalyzing intraday factors...")
        for i, factor_name in enumerate(intraday_factor_names):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(intraday_factor_names)}")
            
            try:
                r = analyze_factor(
                    intraday_df, price_df, factor_name, args.test_start,
                    FILTER_THRESHOLDS['n_quantiles'],
                    FILTER_THRESHOLDS['long_quantile'],
                    FILTER_THRESHOLDS['short_quantile']
                )
                if r:
                    r['type'] = 'Intraday'
                    results[factor_name] = r
            except Exception as e:
                print(f"  Error analyzing {factor_name}: {e}")
    
    print(f"\nValid factors analyzed: {len(results)}")
    
    # 7. 保存全部结果（对 direction=-1 的因子正向化 IC 相关指标）
    all_rows = []
    for name in sorted(results.keys(), key=lambda x: results[x]['sharpe'], reverse=True):
        r = results[name]
        d = r['direction']

        # 如果 direction = -1，将 IC 相关指标取反（正向化）
        # Sharpe 和收益指标保持原值（反映实际策略表现）
        row = {
            'factor': name,
            'type': r['type'],
            'direction': d,
            'ic_mean': r['ic_mean'] * d,           # IC 正向化
            'ic_std': r['ic_std'],                  # 标准差不变
            'ic_ir': r['ic_ir'] * d,               # ICIR 正向化
            'ic_positive_ratio': r['ic_positive_ratio'] if d == 1 else 1 - r['ic_positive_ratio'],
            'sharpe': r['sharpe'],
            'long_sharpe': r['long_sharpe'],
            'short_sharpe': r['short_sharpe'],
            'ann_ret': r['ann_ret'],
            'long_ann_ret': r['long_ann_ret'],
            'short_ann_ret': r['short_ann_ret'],
            'max_dd': r['max_dd'],
            'final_nav': r['final_nav'],
            'n_days': r['n_days'],
            'avg_valid_count': r['avg_valid_count'],
        }
        # 添加年度收益
        for yr, ret in r['yearly_ret'].items():
            row[f'ret_{yr}'] = ret
        all_rows.append(row)
    
    df_results = pd.DataFrame(all_rows)
    all_results_path = output_dir / 'all_factor_results.csv'
    df_results.to_csv(all_results_path, index=False)
    print(f"\nSaved all results: {all_results_path}")
    
    # 8. 筛选合格因子
    qualified = filter_qualified_factors(results, FILTER_THRESHOLDS)
    print(f"\nQualified factors (|IC|>={FILTER_THRESHOLDS['min_ic_abs']}, Sharpe>={FILTER_THRESHOLDS['min_sharpe']}): {len(qualified)}")

    # 9. 基于相关性去重
    qualified_dedup, removed_pairs = dedup_factors_by_correlation(
        qualified, results, daily_df, intraday_df, DEDUP_CORR_THRESHOLD
    )
    if removed_pairs:
        print(f"\nRemoved {len(removed_pairs)} duplicate factors (corr > {DEDUP_CORR_THRESHOLD}):")
        for removed, kept, corr in removed_pairs:
            print(f"  - {removed} (corr={corr:.4f} with {kept})")
        print(f"After dedup: {len(qualified_dedup)} factors")

    if qualified_dedup:
        df_qualified = df_results[df_results['factor'].isin(qualified_dedup)].copy()
        qualified_path = output_dir / 'qualified_factors.csv'
        df_qualified.to_csv(qualified_path, index=False)
        print(f"Saved qualified factors: {qualified_path}")
    
    # 9. 打印结果摘要
    print("\n" + "=" * 130)
    print("ALL FACTORS (sorted by Sharpe)")
    print("=" * 130)
    print(f"{'#':<3} {'Factor':<30} {'Type':<10} {'Dir':>4} {'IC':>8} {'ICIR':>7} {'IC+%':>6} {'Sharpe':>8} {'Long':>7} {'Short':>7} {'MaxDD':>7} {'Pass'}")
    print("-" * 130)

    for i, row in enumerate(all_rows[:50]):  # 前50个
        passed = "✓" if row['factor'] in qualified_dedup else " "
        dir_str = "+" if row['direction'] == 1 else "-"
        print(f"{i+1:<3} {row['factor']:<30} {row['type']:<10} {dir_str:>4} "
              f"{row['ic_mean']:>+.4f} {row['ic_ir']:>+.2f} {row['ic_positive_ratio']*100:>5.1f}% "
              f"{row['sharpe']:>8.2f} {row['long_sharpe']:>7.2f} {row['short_sharpe']:>7.2f} "
              f"{row['max_dd']*100:>6.1f}% {passed}")

    if len(all_rows) > 50:
        print(f"\n... and {len(all_rows) - 50} more factors")

    # 11. 打印合格因子详情
    if qualified_dedup:
        print("\n" + "=" * 100)
        print(f"QUALIFIED FACTORS (after dedup): {len(qualified_dedup)}")
        print("=" * 100)

        # 按类型分组
        daily_qualified = [f for f in qualified_dedup if results[f]['type'] == 'Daily']
        intraday_qualified = [f for f in qualified_dedup if results[f]['type'] == 'Intraday']

        print(f"\nDaily factors ({len(daily_qualified)}):")
        for name in sorted(daily_qualified, key=lambda x: results[x]['sharpe'], reverse=True):
            r = results[name]
            dir_str = "(+)" if r['direction'] == 1 else "(-)"
            print(f"  {name:<30} {dir_str} IC={r['ic_mean']:>+.4f}, Sharpe={r['sharpe']:>6.2f}, MaxDD={r['max_dd']*100:>5.1f}%")

        print(f"\nIntraday factors ({len(intraday_qualified)}):")
        for name in sorted(intraday_qualified, key=lambda x: results[x]['sharpe'], reverse=True):
            r = results[name]
            dir_str = "(+)" if r['direction'] == 1 else "(-)"
            print(f"  {name:<30} {dir_str} IC={r['ic_mean']:>+.4f}, Sharpe={r['sharpe']:>6.2f}, MaxDD={r['max_dd']*100:>5.1f}%")

    # 12. 统计摘要
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    daily_results = [r for r in results.values() if r['type'] == 'Daily']
    intraday_results = [r for r in results.values() if r['type'] == 'Intraday']

    if daily_results:
        daily_qualified_count = len([f for f in qualified_dedup if results[f]['type'] == 'Daily'])
        print(f"\nDaily Factors: {len(daily_results)} analyzed, {daily_qualified_count} qualified")
        print(f"  Avg IC:     {np.mean([r['ic_mean'] for r in daily_results]):+.4f}")
        print(f"  Avg Sharpe: {np.mean([r['sharpe'] for r in daily_results]):.2f}")

    if intraday_results:
        intraday_qualified_count = len([f for f in qualified_dedup if results[f]['type'] == 'Intraday'])
        print(f"\nIntraday Factors: {len(intraday_results)} analyzed, {intraday_qualified_count} qualified")
        print(f"  Avg IC:     {np.mean([r['ic_mean'] for r in intraday_results]):+.4f}")
        print(f"  Avg Sharpe: {np.mean([r['sharpe'] for r in intraday_results]):.2f}")

    print(f"\nTotal: {len(results)} factors analyzed, {len(qualified)} passed filter, {len(qualified_dedup)} after dedup")
    print(f"Qualification rate: {len(qualified_dedup)/len(results)*100:.1f}%")

    print("\n" + "=" * 80)
    print("OUTPUT FILES:")
    print(f"  1. {all_results_path} - All factor results")
    print(f"  2. {qualified_path if qualified_dedup else 'N/A'} - Qualified factors (after dedup)")
    print(f"  3. {thresholds_path} - Daily winsorize thresholds")
    print("=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()