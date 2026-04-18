"""
Forward Test: 2026年1-2月纯样本外测试

从1分钟数据计算8个因子，复用 crypto_strategy.py 的回测引擎。
产出: output/forward_test_performance.png (与 strategy_performance.png 同格式)

前提:
    python download_forward_data.py  # 先下载数据

使用:
    python forward_test.py
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, Optional
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 复用策略模块
from crypto_strategy import (
    FACTOR_CONFIG, Backtester, BacktestResult,
    precompute_funding_costs,
    precompute_expected_funding_rate,
    zscore, daily_winsorize,
)

# ============== 配置 ==============
KLINE_DIR = './futures_data_1m_forward'
FUNDING_DIR = './funding_rates_forward'
OUTPUT_DIR = './output'

# 策略参数 (与主策略完全一致)
LONG_PCT = 0.2
SHORT_PCT = 0.2
WEIGHT_METHOD = 'sqrt_volume'
LEVERAGE = 1.0
TAKER_FEE = 0.0005
IMPACT_COEF = 0.12
AUM = 1_000_000
MAX_PARTICIPATION = 0.05
LIQUIDITY_FILTER = 0.4

# 资金费率干预（信号调整 + 费率扣除，After/Before Trading Cost 一致生效）
ENABLE_FUNDING = True        # 资金费率干预总开关
COST_ADJ_PENALTY = 200       # 成本惩罚系数
COST_ADJ_WINDOW = 14         # 预期费率滚动窗口（EWM span）
COST_ADJ_METHOD = 'ewm'      # 预期费率计算方法（ewm/sma）


# ============== 因子计算 ==============

def compute_intraday_factors_one_day(group: pd.DataFrame, symbol: str, date,
                                      btc_ret: Optional[np.ndarray] = None) -> Optional[Dict]:
    """计算所有合格池中的 intraday 因子（超集，由 build_factor_matrix 按 FACTOR_CONFIG 筛选）"""
    result = {'symbol': symbol, 'date': date}

    ret = group['ret'].fillna(0).values
    if len(ret) < 60:
        return None

    prices = group['close'].values
    volumes = group['volume'].values if 'volume' in group.columns else None
    n = len(ret)

    # --- 1. trade_size_skew ---
    if 'avg_trade_size' in group.columns:
        trade_size = group['avg_trade_size'].dropna().values
        trade_size = trade_size[trade_size > 0]
        if len(trade_size) > 10:
            result['trade_size_skew'] = float(stats.skew(trade_size))

    # --- 2. max_buy_pressure_1h, taker_imbalance_std ---
    if 'taker_imb' in group.columns:
        taker_imb = group['taker_imb'].dropna().values
        if len(taker_imb) > 10:
            result['taker_imbalance_std'] = float(taker_imb.std())
        if len(taker_imb) >= 60:
            rolling = np.convolve(taker_imb, np.ones(60), mode='valid')
            result['max_buy_pressure_1h'] = float(rolling.max())

    # --- 3. early_main_diff ---
    if len(ret) >= 60:
        early_ret = ret[:30].sum()
        main_ret = ret[30:].sum()
        result['early_main_diff'] = float(early_ret - main_ret)

    # --- 4. max_drawdown_intra ---
    if len(prices) > 10:
        cum_max = np.maximum.accumulate(prices)
        drawdown = (prices - cum_max) / (cum_max + 1e-8)
        result['max_drawdown_intra'] = float(drawdown.min())

    # --- 5. info_entropy ---
    if len(ret) >= 60:
        n_periods = 12
        period_len = n // n_periods
        period_abs_ret = []
        for i in range(n_periods):
            start = i * period_len
            end = start + period_len if i < n_periods - 1 else n
            period_abs_ret.append(np.abs(ret[start:end]).sum())
        period_abs_ret = np.array(period_abs_ret)
        total_abs = period_abs_ret.sum()
        if total_abs > 0:
            prob = period_abs_ret / total_abs
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            result['info_entropy'] = float(entropy)

    # --- 6. amihud_illiq_log, amihud_illiq, illiq_variability ---
    if volumes is not None and len(ret) >= 30:
        vol_n = min(len(ret), len(volumes))
        ret_a, vol_a = ret[:vol_n], volumes[:vol_n]
        valid = vol_a > 0
        if valid.sum() >= 30:
            ret_valid = np.abs(ret_a[valid])
            vol_valid = vol_a[valid]
            illiq_values = ret_valid / vol_valid
            illiq_clipped = np.clip(illiq_values, 0, np.percentile(illiq_values, 99))
            amihud_mean = illiq_clipped.mean()
            result['amihud_illiq'] = float(amihud_mean)
            result['amihud_illiq_log'] = float(np.log(amihud_mean + 1e-10))
            if amihud_mean > 0:
                result['illiq_variability'] = float(illiq_clipped.std() / amihud_mean)

    # --- 7. market_r2 ---
    if btc_ret is not None and len(btc_ret) >= len(ret) and len(ret) >= 60:
        btc_aligned = btc_ret[:len(ret)]
        valid_mask = ~(np.isnan(ret) | np.isnan(btc_aligned))
        valid_count = valid_mask.sum()
        if valid_count >= 30:
            ret_valid = ret[valid_mask]
            btc_valid = btc_aligned[valid_mask]
            btc_var = btc_valid.var()
            if btc_var > 1e-12:
                beta = np.cov(ret_valid, btc_valid)[0, 1] / btc_var
                alpha = ret_valid.mean() - beta * btc_valid.mean()
                residual = ret_valid - alpha - beta * btc_valid
                r_squared = 1 - residual.var() / (ret_valid.var() + 1e-12)
                result['market_r2'] = float(r_squared)

    # --- 8. down_vol_ratio ---
    up_ret = ret[ret > 0]
    down_ret = ret[ret < 0]
    up_vol = np.sqrt((up_ret ** 2).sum()) if len(up_ret) > 0 else 0
    down_vol = np.sqrt((down_ret ** 2).sum()) if len(down_ret) > 0 else 0
    total_vol = up_vol + down_vol
    if total_vol > 0:
        result['down_vol_ratio'] = float(down_vol / total_vol)

    # --- 9. high_vol_up_reversal, high_vol_down_momentum ---
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        up_mask = ret > 0
        down_mask = ret < 0
        vol_median = np.median(vol_aligned)
        high_vol_mask = vol_aligned > vol_median

        if (up_mask & high_vol_mask).sum() > 5:
            high_vol_up_ret = ret[up_mask & high_vol_mask].sum()
            result['high_vol_up_reversal'] = float(-high_vol_up_ret)

        if (down_mask & high_vol_mask).sum() > 5:
            high_vol_down_ret = ret[down_mask & high_vol_mask].sum()
            result['high_vol_down_momentum'] = float(high_vol_down_ret)

    # --- 10. min_1h_ret ---
    if len(ret) >= 60:
        rolling_1h = np.convolve(ret, np.ones(60), mode='valid')
        result['min_1h_ret'] = float(rolling_1h.min())

    # --- 11. taker_residual_std ---
    if 'taker_imb' in group.columns:
        taker_imb = group['taker_imb'].dropna().values
        if len(taker_imb) >= n:
            taker_aligned = taker_imb[:n]
            valid_tr = ~(np.isnan(ret) | np.isnan(taker_aligned))
            if valid_tr.sum() > 30:
                taker_valid = taker_aligned[valid_tr]
                ret_valid_tr = ret[valid_tr]
                if ret_valid_tr.std() > 0:
                    beta_t = np.cov(taker_valid, ret_valid_tr)[0, 1] / (ret_valid_tr.var() + 1e-8)
                    residual_t = taker_valid - beta_t * ret_valid_tr
                    result['taker_residual_std'] = float(residual_t.std())

    # --- 12. gamma_down ---
    if 'taker_imb' in group.columns and volumes is not None:
        taker_imb = group['taker_imb'].dropna().values
        if len(taker_imb) >= n and len(volumes) >= n:
            bar_size = 5
            n_bars = n // bar_size
            if n_bars >= 12:
                ret_5m = np.array([ret[i*bar_size:(i+1)*bar_size].sum() for i in range(n_bars)])
                taker_5m = np.array([taker_imb[i*bar_size:(i+1)*bar_size].mean() for i in range(n_bars)])
                mf = taker_5m
                buy_mask = mf > 0
                sell_mask = mf < 0
                if buy_mask.sum() >= 5 and sell_mask.sum() >= 5:
                    mf_sell = np.abs(mf[sell_mask])
                    ret_sell = ret_5m[sell_mask]
                    if mf_sell.std() > 0:
                        gamma_down_val = np.cov(ret_sell, mf_sell)[0, 1] / (mf_sell.var() + 1e-10)
                        result['gamma_down'] = float(gamma_down_val)

    # --- 13. vol_cv, vol_gini ---
    if volumes is not None and len(volumes) >= n:
        vol_aligned = volumes[:n]
        vol_mean = vol_aligned.mean()
        vol_std = vol_aligned.std()
        if vol_mean > 0:
            result['vol_cv'] = float(vol_std / vol_mean)
            # Gini coefficient
            vol_sorted = np.sort(vol_aligned)
            n_vol = len(vol_sorted)
            total_vol_sum = vol_sorted.sum()
            if total_vol_sum > 0:
                idx = np.arange(1, n_vol + 1)
                gini = (2 * np.sum(idx * vol_sorted) - (n_vol + 1) * total_vol_sum) / (n_vol * total_vol_sum)
                result['vol_gini'] = float(gini)

    return result


def load_and_compute_factors(kline_dir=None) -> tuple:
    """加载1分钟数据, 计算所有8个因子, 聚合日度OHLCV

    Args:
        kline_dir: K线数据目录，默认使用 KLINE_DIR 常量

    Returns:
        (intraday_factors_df, daily_ohlcv_df)
    """
    kline_dir = kline_dir or KLINE_DIR
    data_path = Path(kline_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"K线数据目录不存在: {kline_dir}, 请先运行 download_forward_data.py")

    symbol_files = {}
    from download_forward_data import QUALIFIED_POOL
    for symbol in QUALIFIED_POOL:
        fp = data_path / f"{symbol}.parquet"
        if fp.exists():
            symbol_files[symbol] = fp

    print(f"Found {len(symbol_files)} symbol files in {kline_dir}")
    if len(symbol_files) == 0:
        raise ValueError("No data files found!")

    # --- 加载BTC分钟收益率 ---
    btc_returns = None
    btc_path = data_path / "BTCUSDT.parquet"
    if btc_path.exists():
        print("Loading BTC minute returns for market_r2...")
        btc_df = pd.read_parquet(btc_path)
        btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])
        btc_df['btc_ret'] = btc_df['close'].pct_change()
        btc_returns = btc_df[['open_time', 'btc_ret']].set_index('open_time')
        print(f"  BTC: {len(btc_returns):,} minute bars")
        del btc_df
        gc.collect()

    # --- 逐symbol处理 ---
    factors_list = []
    daily_list = []
    processed = 0

    for symbol, file_path in symbol_files.items():
        processed += 1
        if processed % 50 == 0:
            print(f"  Processing {processed}/{len(symbol_files)}: {symbol}")

        try:
            df = pd.read_parquet(file_path)
            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            df['open_time'] = pd.to_datetime(df['open_time'])
            df['date'] = df['open_time'].dt.date

            # 数据预处理 (复刻 factor_mining.py:1606-1612)
            df['ret'] = df['close'].pct_change()
            df['avg_trade_size'] = df['volume'] / (df['trades'] + 1e-8)
            if 'taker_buy_volume' in df.columns:
                df['taker_imb'] = (2 * df['taker_buy_volume'] - df['volume']) / (df['volume'] + 1e-8)

            # 合并BTC收益率
            if btc_returns is not None and symbol != 'BTCUSDT':
                df = df.set_index('open_time').join(btc_returns, how='left').reset_index()
            else:
                df['btc_ret'] = np.nan

            # 日度OHLCV聚合
            daily = df.groupby('date').agg({
                'symbol': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'taker_buy_volume': 'sum',
            }).reset_index()
            daily_list.append(daily)

            # 按日计算intraday因子
            for date, group in df.groupby('date'):
                if len(group) < 60:
                    continue
                btc_ret_day = group['btc_ret'].values if 'btc_ret' in group.columns else None
                row = compute_intraday_factors_one_day(group, symbol, date, btc_ret_day)
                if row:
                    factors_list.append(row)

            del df
            gc.collect()

        except Exception as e:
            print(f"  Error: {symbol} - {e}")

    print(f"\nProcessed {processed} symbols")
    print(f"Intraday factor rows: {len(factors_list):,}")

    intraday_df = pd.DataFrame(factors_list) if factors_list else pd.DataFrame()
    daily_df = pd.concat(daily_list, ignore_index=True) if daily_list else pd.DataFrame()

    return intraday_df, daily_df


def compute_lower_shadow(daily_df: pd.DataFrame) -> pd.DataFrame:
    """计算daily因子: lower_shadow"""
    daily_df = daily_df.copy()
    body_low = np.minimum(
        pd.to_numeric(daily_df['open'], errors='coerce'),
        pd.to_numeric(daily_df['close'], errors='coerce')
    )
    low = pd.to_numeric(daily_df['low'], errors='coerce')
    high = pd.to_numeric(daily_df['high'], errors='coerce')
    daily_df['lower_shadow'] = (body_low - low) / (high - low + 1e-8)
    return daily_df[['date', 'symbol', 'lower_shadow']]


def build_factor_matrix(intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """构建因子宽表字典 {factor_name: DataFrame(date x symbol)}"""
    ls_df = compute_lower_shadow(daily_df)

    if len(intraday_df) > 0:
        intraday_df['date'] = pd.to_datetime(intraday_df['date'])
    ls_df['date'] = pd.to_datetime(ls_df['date'])

    factors = {}
    factor_names = list(FACTOR_CONFIG.keys())

    for fname in factor_names:
        if fname == 'lower_shadow':
            src = ls_df
        else:
            src = intraday_df

        if fname not in src.columns:
            print(f"  Warning: factor '{fname}' not found in data")
            continue

        wide = src.pivot_table(index='date', columns='symbol', values=fname, aggfunc='first')
        factors[fname] = wide

    print(f"\nFactor matrix built: {list(factors.keys())}")
    for fname, fdf in factors.items():
        coverage = fdf.notna().sum().sum() / (fdf.shape[0] * fdf.shape[1]) * 100
        print(f"  {fname}: {fdf.shape[0]} days x {fdf.shape[1]} symbols, coverage={coverage:.1f}%")

    return factors



def build_returns_and_volume(daily_df: pd.DataFrame) -> tuple:
    """从日度OHLCV构建前向收益率、quote_volume、close_price宽表"""
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    for col in ['open', 'close', 'quote_volume']:
        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')

    open_ = daily_df.pivot_table(index='date', columns='symbol', values='open', aggfunc='first')
    close = daily_df.pivot_table(index='date', columns='symbol', values='close', aggfunc='last')
    quote_volume = daily_df.pivot_table(index='date', columns='symbol', values='quote_volume', aggfunc='sum')

    open_ret = open_.pct_change()
    fwd_ret = daily_winsorize(open_ret.shift(-2), 1, 99)

    print(f"\nReturns: {len(fwd_ret)} days x {len(fwd_ret.columns)} symbols")
    print(f"Date range: {fwd_ret.index.min().date()} ~ {fwd_ret.index.max().date()}")

    return fwd_ret, quote_volume, close


# ============== 主流程 ==============

def run_forward_test():
    """运行forward test主流程"""
    print("=" * 70)
    print("FORWARD TEST: 2026-01 ~ 2026-02")
    if ENABLE_FUNDING:
        print(f"  Funding intervention: penalty={COST_ADJ_PENALTY}, window={COST_ADJ_WINDOW}d")
    else:
        print(f"  Funding intervention: OFF")
    print("=" * 70)

    # 1. 加载数据并计算因子
    print("\n[1/4] Loading data and computing factors...")
    intraday_df, daily_df = load_and_compute_factors()

    if len(intraday_df) == 0 or len(daily_df) == 0:
        print("ERROR: No data available. Run download_forward_data.py first.")
        return None

    # 2. 构建因子矩阵
    print("\n[2/4] Building factor matrix...")
    raw_factors = build_factor_matrix(intraday_df, daily_df)

    if len(raw_factors) == 0:
        print("ERROR: No factors computed.")
        return None

    factors = {k: zscore(v) for k, v in raw_factors.items()}

    # 3. 等权组合信号 + 过滤
    print("\n[3/4] Generating combined signal...")
    common_dates = sorted(set.intersection(*[set(f.index) for f in factors.values()]))
    common_symbols = sorted(set.intersection(*[set(f.columns) for f in factors.values()]))

    print(f"  Common dates: {len(common_dates)}, symbols: {len(common_symbols)}")

    if len(common_dates) < 5 or len(common_symbols) < 20:
        print("ERROR: Insufficient data overlap.")
        return None

    combined_signal = sum(
        FACTOR_CONFIG[name].direction * f.reindex(index=common_dates, columns=common_symbols).fillna(0)
        for name, f in factors.items()
    ) / len(factors)

    # 资金费率干预（信号调整 + 费率扣除）
    funding_kwargs = {}
    if ENABLE_FUNDING:
        print(f"  Applying cost adjustment (method={COST_ADJ_METHOD}, "
              f"penalty={COST_ADJ_PENALTY}, window={COST_ADJ_WINDOW}d)...")
        efr_wide = precompute_expected_funding_rate(
            funding_dir=['./funding_rates', FUNDING_DIR],
            window=COST_ADJ_WINDOW,
            method=COST_ADJ_METHOD,
            verbose=True,
        )
        if not efr_wide.empty:
            efr_aligned = efr_wide.reindex(
                index=combined_signal.index, columns=combined_signal.columns
            ).fillna(0)
            signal_raw = combined_signal.copy()
            combined_signal = combined_signal - COST_ADJ_PENALTY * efr_aligned
            combined_signal[signal_raw.isna()] = np.nan
            adj_nonzero = (efr_aligned != 0).sum().sum()
            total_slots = combined_signal.shape[0] * combined_signal.shape[1]
            print(f"  [CostAdj] Adjusted {adj_nonzero}/{total_slots} entries "
                  f"({adj_nonzero/total_slots*100:.1f}%)")

        funding_dir_path = Path(FUNDING_DIR)
        if funding_dir_path.exists() and any(funding_dir_path.glob("*.parquet")):
            print("  Loading funding rate costs...")
            fa = precompute_funding_costs(funding_dir=str(FUNDING_DIR), verbose=True)
            funding_kwargs = dict(funding_adj=fa)

    # 4. 收益率 + 回测
    print("\n[4/4] Building returns and running backtest...")
    fwd_ret, quote_volume, close_price = build_returns_and_volume(daily_df)

    bt = Backtester(fwd_ret, quote_volume, close_price)

    # After cost
    result = bt.run(
        combined_signal,
        long_pct=LONG_PCT, short_pct=SHORT_PCT,
        weight_method=WEIGHT_METHOD, leverage=LEVERAGE,
        enable_cost=True, taker_fee=TAKER_FEE,
        impact_coef=IMPACT_COEF, aum=AUM,
        max_participation=MAX_PARTICIPATION,
        liquidity_filter=LIQUIDITY_FILTER,
        **funding_kwargs,
    )

    # Before cost (不含交易成本，资金费率干预与 after cost 一致)
    benchmark = bt.run(
        combined_signal,
        long_pct=LONG_PCT, short_pct=SHORT_PCT,
        weight_method=WEIGHT_METHOD, leverage=LEVERAGE,
        enable_cost=False, aum=AUM,
        max_participation=MAX_PARTICIPATION,
        liquidity_filter=LIQUIDITY_FILTER,
        **funding_kwargs,
    )

    # 打印结果
    print_results(result, benchmark)

    # 绘图
    plot_results(result, benchmark)

    return result


def print_results(result: BacktestResult, benchmark: BacktestResult):
    """打印回测结果"""
    print("\n" + "=" * 70)
    print("FORWARD TEST RESULTS")
    print("=" * 70)

    n_days = len(result.daily_returns)
    print(f"{'Period':<25}{result.daily_returns.index.min().strftime('%Y-%m-%d')} ~ "
          f"{result.daily_returns.index.max().strftime('%Y-%m-%d')}")
    print(f"{'Trading Days':<25}{n_days}")

    print(f"\n{'Metric':<25}{'After Trading Cost':>15}{'Before Trading Cost':>15}")
    print("-" * 55)
    print(f"{'Ann. Return':<25}{result.annualized_return:>14.1%}{benchmark.annualized_return:>14.1%}")
    print(f"{'Sharpe Ratio':<25}{result.sharpe_ratio:>15.2f}{benchmark.sharpe_ratio:>15.2f}")
    print(f"{'Max Drawdown':<25}{result.max_drawdown:>14.1%}{benchmark.max_drawdown:>14.1%}")
    print(f"{'Net Value':<25}{result.total_net_value:>14.4f}x{benchmark.total_net_value:>13.4f}x")
    print(f"{'Win Rate':<25}{result.win_rate:>14.1%}{benchmark.win_rate:>14.1%}")

    print(f"\n--- Cost Analysis ---")
    print(f"{'Total Cost':<25}{result.total_cost:>14.4f}")
    print(f"{'Avg Daily Turnover':<25}{result.avg_daily_turnover:>14.2%}")
    print(f"{'Avg Utilization':<25}{result.avg_utilization:>14.1%}")

    # 月度收益
    print(f"\n--- Monthly Returns ---")
    monthly = result.daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    for date, ret in monthly.items():
        print(f"  {date.strftime('%Y-%m')}: {ret:>8.2%}")


def plot_results(result: BacktestResult, benchmark: BacktestResult):
    """绘制净值曲线 + 回撤 (与 strategy_performance.png 同格式)"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                              gridspec_kw={'hspace': 0.15})
    fig.suptitle(f"Forward Test Performance (Sharpe: {result.sharpe_ratio:.2f}, Leverage: {LEVERAGE}x)",
                 fontsize=16, fontweight='bold', y=0.95)

    cum = result.cumulative_returns
    bm_cum = benchmark.cumulative_returns

    # ==================== 上图：累计收益 ====================
    ax1 = axes[0]

    ax1.plot(cum.index, cum.values, color='#1f77b4', linewidth=2.5,
             label=f'After Trading Cost (Sharpe: {result.sharpe_ratio:.2f})')
    ax1.plot(bm_cum.index, bm_cum.values, color='#d62728', linewidth=2, alpha=0.9,
             label=f'Before Trading Cost (Sharpe: {benchmark.sharpe_ratio:.2f})')

    ax1.set_ylabel('Normalized Equity (Base=1.0)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(cum.index[0], cum.index[-1])
    ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.text(cum.index[0], 1, ' 1.0', fontsize=9, va='center', ha='left', color='black', alpha=0.7)

    # 统计信息框
    stats_text = (
        f"After Trading Cost:\n"
        f"  Ann.Ret: {result.annualized_return:>7.1%}\n"
        f"  MaxDD:   {result.max_drawdown:>7.1%}\n"
        f"  Sharpe:  {result.sharpe_ratio:>7.2f}\n"
        f"  WinRate: {result.win_rate:>7.1%}\n"
        f"Before Trading Cost:\n"
        f"  Ann.Ret: {benchmark.annualized_return:>7.1%}\n"
        f"  MaxDD:   {benchmark.max_drawdown:>7.1%}\n"
        f"  Sharpe:  {benchmark.sharpe_ratio:>7.2f}\n"
        f"Cost:\n"
        f"  Total:   {result.total_cost:>7.1%}\n"
        f"  Turnover:{result.avg_daily_turnover:>7.1%}"
    )

    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.85, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', bbox=props)

    # ==================== 下图：回撤 ====================
    ax2 = axes[1]

    cum_clean = cum.dropna()
    bm_cum_clean = bm_cum.dropna()
    bm_dd = None

    if len(cum_clean) > 0:
        rm = np.maximum.accumulate(cum_clean.values)
        dd = (cum_clean.values - rm) / (rm + 1e-8) * 100
        ax2.fill_between(cum_clean.index, 0, dd, color='#1f77b4', alpha=0.6, label='After Trading Cost')

    if len(bm_cum_clean) > 0:
        bm_rm = np.maximum.accumulate(bm_cum_clean.values)
        bm_dd = (bm_cum_clean.values - bm_rm) / (bm_rm + 1e-8) * 100
        ax2.fill_between(bm_cum_clean.index, 0, bm_dd, color='#d62728', alpha=0.3, label='Before Trading Cost')

    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(cum.index[0], cum.index[-1])
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')

    min_dd = dd.min() if len(cum_clean) > 0 else -10
    if bm_dd is not None:
        min_dd = min(min_dd, bm_dd.min())
    ax2.set_ylim(min_dd * 1.1, 0.5)

    plt.tight_layout()

    out_path = Path(OUTPUT_DIR) / 'forward_test_performance.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    try:
        result = run_forward_test()
        if result is None:
            print("\nForward test failed. Check data availability.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
