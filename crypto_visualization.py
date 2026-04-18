"""
多因子策略可视化与执行脚本

执行逻辑：
1. 从CSV文件加载预计算因子
2. 运行策略回测
3. 生成可视化报告
4. VIP等级 × AUM组合对比分析
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from crypto_strategy import (
    BacktestResult, Backtester, FactorLoader,
    FactorAnalyzer, FACTOR_CONFIG,
    load_returns_from_price,
    precompute_funding_costs,
    precompute_expected_funding_rate,
)


# ============================================================
# VIP等级 × AUM 配置
# ============================================================

# VIP等级Taker费率 (Binance USDT-M Futures)
VIP_FEES = {
    'VIP0': 0.0005,    # 5.0 bps
    'VIP6': 0.00025,   # 2.5 bps
    'VIP9': 0.00017,   # 1.7 bps
}

# AUM配置
AUM_LEVELS = {
    '$1M': 1_000_000,
    '$5M': 5_000_000,
    '$10M': 10_000_000,
}


# ============================================================
# 可视化函数
# ============================================================

def plot_backtest_results(result: BacktestResult, benchmark_result: BacktestResult = None,
                          title: str = "Strategy Performance", save_path: Optional[str] = None,
                          show_cost_metrics: bool = True, market_index: pd.Series = None,
                          oos_start: str = None, oos_end: str = None):
    """绘制回测结果（样本内和样本外通用）"""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                              gridspec_kw={'hspace': 0.15})
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    cum = result.cumulative_returns
    is_oos = oos_start is not None or oos_end is not None

    # ==================== 上图：累计收益 ====================
    ax1 = axes[0]

    if market_index is not None:
        ax1_right = ax1.twinx()
        market_index_smooth = market_index.rolling(window=20, min_periods=1).mean()
        ax1_right.plot(market_index_smooth.index, market_index_smooth.values,
                       color='gray', linewidth=1, alpha=0.35, label='Crypto Market Index (MA20)')
        ax1_right.set_ylabel('Market Index', fontsize=10, color='gray')
        ax1_right.tick_params(axis='y', labelcolor='gray')
        ax1_right.legend(loc='upper right', fontsize=9)

    ax1.plot(cum.index, cum.values, color='#1f77b4', linewidth=2.5,
             label=f'After Trading Cost (Sharpe: {result.sharpe_ratio:.2f})')

    bm_dd = None
    if benchmark_result:
        bm_cum = benchmark_result.cumulative_returns
        ax1.plot(bm_cum.index, bm_cum.values, color='#d62728', linewidth=2,
                 alpha=0.9, label=f'Before Trading Cost (Sharpe: {benchmark_result.sharpe_ratio:.2f})')

    ax1.set_ylabel('Normalized Equity (Base=1.0)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(cum.index[0], cum.index[-1])
    ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.text(cum.index[0], 1, ' 1.0', fontsize=9, va='center', ha='left', color='black', alpha=0.7)

    if is_oos:
        period_info = f"Period: {oos_start} ~ {oos_end}\n" if oos_start and oos_end else ""
        stats_text = (
            f"OUT-OF-SAMPLE\n"
            f"{period_info}"
            f"Days: {len(cum)}\n"
            f"\nAfter Trading Cost:\n"
        )
    else:
        stats_text = "After Trading Cost:\n"

    stats_text += (
        f"  Ann.Ret: {result.annualized_return:>7.1%}\n"
        f"  MaxDD:   {result.max_drawdown:>7.1%}\n"
        f"  Sharpe:  {result.sharpe_ratio:>7.2f}\n"
        f"  WinRate: {result.win_rate:>7.1%}"
    )

    if benchmark_result:
        stats_text += (
            f"\nBefore Trading Cost:\n"
            f"  Ann.Ret: {benchmark_result.annualized_return:>7.1%}\n"
            f"  MaxDD:   {benchmark_result.max_drawdown:>7.1%}\n"
            f"  Sharpe:  {benchmark_result.sharpe_ratio:>7.2f}"
        )

    if show_cost_metrics and result.total_cost > 0:
        stats_text += (
            f"\nCost:\n"
            f"  Total:   {result.total_cost:>7.1%}\n"
            f"  Turnover:{result.avg_daily_turnover:>7.1%}"
        )

    box_color = 'lightcyan' if is_oos else 'wheat'
    box_alpha = 0.9 if is_oos else 0.8
    props = dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=box_alpha)
    ax1.text(0.02, 0.85, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', bbox=props)

    # ==================== 下图：回撤 ====================
    ax2 = axes[1]

    dd = None
    cum_clean = cum.dropna()
    if len(cum_clean) > 0:
        rm = np.maximum.accumulate(cum_clean.values)
        dd = (cum_clean.values - rm) / (rm + 1e-8) * 100
        ax2.fill_between(cum_clean.index, 0, dd, color='#1f77b4', alpha=0.6, label='After Trading Cost')

    if benchmark_result:
        bm_cum_clean = benchmark_result.cumulative_returns.dropna()
        if len(bm_cum_clean) > 0:
            bm_rm = np.maximum.accumulate(bm_cum_clean.values)
            bm_dd = (bm_cum_clean.values - bm_rm) / (bm_rm + 1e-8) * 100
            ax2.fill_between(bm_cum_clean.index, 0, bm_dd, color='#d62728', alpha=0.3, label='Before Trading Cost')

    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(cum.index[0], cum.index[-1])
    dd_title = 'Out-of-Sample Drawdown' if is_oos else 'Drawdown Comparison'
    ax2.set_title(dd_title, fontsize=12, fontweight='bold')

    min_dd = dd.min() if dd is not None else -10
    if bm_dd is not None:
        min_dd = min(min_dd, bm_dd.min())
    ax2.set_ylim(min_dd * 1.1, 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_out_of_sample_results(result: BacktestResult, benchmark_result: BacktestResult = None,
                               title: str = "Out-of-Sample Performance", save_path: Optional[str] = None,
                               show_cost_metrics: bool = True, market_index: pd.Series = None,
                               oos_start: str = None, oos_end: str = None):
    """向后兼容的样本外绘图函数，委托给 plot_backtest_results"""
    plot_backtest_results(result, benchmark_result=benchmark_result, title=title,
                          save_path=save_path, show_cost_metrics=show_cost_metrics,
                          market_index=market_index, oos_start=oos_start, oos_end=oos_end)


def plot_performance_summary(result: BacktestResult, benchmark_result: BacktestResult = None,
                              title: str = "Performance Summary", save_path: Optional[str] = None,
                              show_cost_metrics: bool = True):
    """绘制收益分布和统计表格"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    daily = result.daily_returns
    
    # ==================== 左图：收益分布 ====================
    ax1 = axes[0]
    ax1.hist(daily.dropna(), bins=50, alpha=0.7, color='#1f77b4', edgecolor='white')
    ax1.axvline(x=daily.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily.mean():.2%}')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_title('Daily Returns Distribution', fontweight='bold')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ==================== 右图：指标表格 ====================
    ax2 = axes[1]
    ax2.axis('off')
    
    metrics = [
        ['Net Value', f'{result.total_net_value:.2f}x'],
        ['Ann. Return', f'{result.annualized_return:.1%}'],
        ['Ann. Volatility', f'{result.annualized_volatility:.1%}'],
        ['Sharpe Ratio', f'{result.sharpe_ratio:.2f}'],
        ['Max Drawdown', f'{result.max_drawdown:.1%}'],
        ['Win Rate', f'{result.win_rate:.1%}'],
        ['Profit/Loss Ratio', f'{result.profit_loss_ratio:.2f}'],
    ]
    
    if show_cost_metrics and result.total_cost > 0:
        metrics.extend([
            ['─' * 15, '─' * 10],
            ['Total Cost', f'{result.total_cost:.1%}'],
            ['Avg Daily Turnover', f'{result.avg_daily_turnover:.1%}'],
            ['Avg Utilization', f'{result.avg_utilization:.1%}'],
        ])
    
    if benchmark_result:
        metrics.extend([
            ['─' * 15, '─' * 10],
            ['Pre-Cost Net Value', f'{benchmark_result.total_net_value:.2f}x'],
            ['Pre-Cost Sharpe', f'{benchmark_result.sharpe_ratio:.2f}'],
        ])
    
    table = ax2.table(cellText=metrics, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='center', colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for j in range(2):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_universe_coverage(data: pd.DataFrame, start_date: str = "2020-01-01",
                           save_path: Optional[str] = None):
    """绘制标的数量热力图"""
    daily_counts = data.groupby(level=0).size()
    daily_counts = daily_counts[daily_counts.index >= start_date]
    monthly_counts = daily_counts.resample('ME').mean()
    
    monthly_df = pd.DataFrame({
        'Year': monthly_counts.index.year,
        'Month': monthly_counts.index.month,
        'Count': monthly_counts.values
    })
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Count')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 热力图
    ax1 = axes[0]
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    im = ax1.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=pivot.values.max())
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] if not pd.isna(pivot.iloc[i, j]) else 0
            ax1.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if val < 30 or val > 70 else 'black', fontweight='bold')
    
    if 2022 in pivot.index.tolist():
        y_pos = pivot.index.tolist().index(2022) - 0.5
        ax1.axhline(y=y_pos, color='blue', linewidth=3, linestyle='--')
        ax1.annotate('Backtest Start →', xy=(11.5, y_pos), color='blue', fontsize=10, 
                    fontweight='bold', ha='right', va='bottom')
    
    ax1.set_title('Monthly Symbol Count', fontweight='bold')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # 时间序列
    ax2 = axes[1]
    ax2.plot(daily_counts.index, daily_counts.values, color='#2E86AB', alpha=0.5, linewidth=0.5)
    ax2.plot(monthly_counts.index, monthly_counts.values, color='#E74C3C', linewidth=2)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Min (50)')
    ax2.axvline(x=pd.Timestamp('2022-01-01'), color='blue', linewidth=2, linestyle='--', 
                label='Backtest Start')
    ax2.fill_between(monthly_counts.index, 0, monthly_counts.values,
                     where=monthly_counts.index < '2022-01-01', color='red', alpha=0.2)
    ax2.fill_between(monthly_counts.index, 0, monthly_counts.values,
                     where=monthly_counts.index >= '2022-01-01', color='green', alpha=0.2)
    ax2.set_title('Symbol Count Over Time', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_tradeable_universe_heatmap(quote_volume: pd.DataFrame, 
                                     start_date: str = "2022-01-01",
                                     liquidity_filter: float = 0.4,
                                     pool_name: str = None,
                                     save_path: Optional[str] = None):
    """
    绘制实际可交易标的数量热力图（经过流动性筛选后）
    
    Args:
        quote_volume: 成交额数据 (date x symbol)，池内的数据
        start_date: 开始日期
        liquidity_filter: 池内quote_volume前N%可交易（默认40%=Q1+Q2）
        pool_name: 池名称，默认使用数据中的币数
        save_path: 保存路径
    """
    # 筛选日期
    qv = quote_volume.loc[quote_volume.index >= start_date].copy()
    
    # 池的总币种数
    pool_size = len(qv.columns)
    pool_label = pool_name if pool_name else f"{pool_size}"
    
    # 计算每日实际可交易标的数
    daily_tradeable = []
    daily_available = []
    
    for date in qv.index:
        row = qv.loc[date].dropna()
        row = row[row > 0]
        
        if len(row) < 10:
            daily_tradeable.append((date, 0))
            daily_available.append((date, 0))
            continue
        
        # 当天池内有数据的币数量
        n_available = len(row)
        daily_available.append((date, n_available))
        
        # 取前40%可交易
        n_tradeable = int(n_available * liquidity_filter)
        daily_tradeable.append((date, n_tradeable))
    
    daily_counts = pd.Series(dict(daily_tradeable))
    daily_avail = pd.Series(dict(daily_available))
    monthly_counts = daily_counts.resample('ME').mean()
    monthly_avail = daily_avail.resample('ME').mean()
    
    monthly_df = pd.DataFrame({
        'Year': monthly_counts.index.year,
        'Month': monthly_counts.index.month,
        'Count': monthly_counts.values
    })
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Count')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Tradeable Universe (Top {liquidity_filter:.0%} Liquidity of {pool_label} Pool)', 
                 fontsize=14, fontweight='bold')
    
    # 热力图
    ax1 = axes[0]
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    im = ax1.imshow(pivot.values, cmap=cmap, aspect='auto', 
                    vmin=pivot.values.min() * 0.8, vmax=pivot.values.max() * 1.1)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] if not pd.isna(pivot.iloc[i, j]) else 0
            text_color = 'white' if val > pivot.values.max() * 0.6 else 'black'
            ax1.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9,
                    color=text_color, fontweight='bold')
    
    ax1.set_title('Monthly Tradeable Symbols', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Symbol Count')
    
    # 时间序列 - 显示可用币数和可交易币数
    ax2 = axes[1]
    ax2.plot(monthly_avail.index, monthly_avail.values, color='#95a5a6', linewidth=2, 
             linestyle='--', label='Available in Pool', alpha=0.7)
    ax2.plot(daily_counts.index, daily_counts.values, color='#2E86AB', alpha=0.3, linewidth=0.5)
    ax2.plot(monthly_counts.index, monthly_counts.values, color='#E74C3C', linewidth=2.5, 
             label=f'Tradeable (Top {liquidity_filter:.0%})')
    
    avg_count = monthly_counts.mean()
    min_count = monthly_counts.min()
    max_count = monthly_counts.max()
    
    ax2.axhline(y=avg_count, color='green', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Avg: {avg_count:.0f}')
    
    ax2.fill_between(monthly_counts.index, 0, monthly_counts.values, color='#2E86AB', alpha=0.2)
    
    ax2.set_title('Tradeable Symbols Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Symbol Count')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    stats_text = f"Period: {start_date} ~ {daily_counts.index[-1].strftime('%Y-%m-%d')}\n"
    stats_text += f"Tradeable - Avg: {avg_count:.0f} | Min: {min_count:.0f} | Max: {max_count:.0f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9)
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right', 
             fontfamily='monospace', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_factor_correlation(corr_matrix: pd.DataFrame, save_path: Optional[str] = None):
    """绘制因子相关性热力图"""
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax, square=True, 
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    ax.set_title('Factor IC Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    
    print("\nHigh correlation pairs (|corr| > 0.5):")
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.5:
                print(f"  {corr_matrix.index[i]} <-> {corr_matrix.columns[j]}: {c:.3f}")


def plot_monthly_returns(daily_returns: pd.Series, save_path: Optional[str] = None):
    """绘制月度收益热力图"""
    monthly = daily_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year, 'Month': monthly.index.month, 'Return': monthly.values
    })
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] if not pd.isna(pivot.iloc[i, j]) else 0
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=9,
                   color='white' if abs(val) > 0.3 else 'black', fontweight='bold')
    
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    yearly = monthly.groupby(monthly.index.year).apply(lambda x: (1+x).prod()-1)
    for i, (year, ret) in enumerate(yearly.items()):
        ax.annotate(f'{ret:.0%}', xy=(12.3, i), fontsize=10, fontweight='bold',
                   color='green' if ret > 0 else 'red')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_factor_analysis(factor_stats: pd.DataFrame, save_path: Optional[str] = None):
    """绘制因子分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # IC柱状图
    ax1 = axes[0]
    if 'ic_mean' in factor_stats.columns:
        sorted_stats = factor_stats.sort_values('ic_mean', ascending=True)
        colors = ['#2E86AB' if FACTOR_CONFIG.get(n, type('', (), {'freq': 'daily'})()).freq == 'daily' 
                  else '#E74C3C' for n in sorted_stats.index]
        ax1.barh(range(len(sorted_stats)), sorted_stats['ic_mean'], color=colors)
        ax1.set_yticks(range(len(sorted_stats)))
        ax1.set_yticklabels(sorted_stats.index, fontsize=9)
        ax1.set_xlabel('IC Mean')
        ax1.set_title('Factor IC (Blue=Daily, Red=Intraday)', fontweight='bold')
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
    
    # IC vs IR散点图
    ax2 = axes[1]
    if 'ic_mean' in factor_stats.columns and 'ic_ir' in factor_stats.columns:
        colors = ['#2E86AB' if FACTOR_CONFIG.get(n, type('', (), {'freq': 'daily'})()).freq == 'daily' 
                  else '#E74C3C' for n in factor_stats.index]
        ax2.scatter(factor_stats['ic_mean'], factor_stats['ic_ir'], c=colors, s=100, alpha=0.7)
        for name, row in factor_stats.iterrows():
            ax2.annotate(name, (row['ic_mean'], row['ic_ir']), fontsize=8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('IC Mean')
        ax2.set_ylabel('IC IR')
        ax2.set_title('Factor IC vs IC_IR', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_comprehensive_report(result: BacktestResult, corr_matrix: pd.DataFrame,
                              data: pd.DataFrame, factor_stats: pd.DataFrame = None,
                              save_path: Optional[str] = None):
    """绘制综合报告"""
    fig = plt.figure(figsize=(18, 12))
    
    cum, daily = result.cumulative_returns, result.daily_returns
    
    # 1. 累计收益
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(cum.index, cum.values, color='#2E86AB', linewidth=2)
    ax1.set_title(f'Cumulative Returns (Sharpe: {result.sharpe_ratio:.2f})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = fig.add_subplot(2, 3, 2)
    rm = np.maximum.accumulate(cum.values)
    dd = (cum.values - rm) / rm
    ax2.fill_between(cum.index, 0, dd, color='#E74C3C', alpha=0.5)
    ax2.set_title(f'Drawdown (Max: {result.max_drawdown:.1%})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 月度热力图
    ax3 = fig.add_subplot(2, 3, 3)
    monthly = daily.resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_df = pd.DataFrame({'Year': monthly.index.year, 'Month': monthly.index.month, 
                               'Return': monthly.values})
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j] if not pd.isna(pivot.iloc[i, j]) else 0
            ax3.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.3 else 'black')
    ax3.set_title('Monthly Returns', fontweight='bold')
    
    # 4. 因子相关性
    ax4 = fig.add_subplot(2, 3, 4)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax4, square=True, cbar_kws={'shrink': 0.6},
                annot_kws={'size': 7})
    ax4.set_title('Factor Correlations', fontweight='bold')
    
    # 5. 标的数量
    ax5 = fig.add_subplot(2, 3, 5)
    daily_counts = data.groupby(level=0).size()
    daily_counts = daily_counts[daily_counts.index >= '2020-01-01']
    monthly_counts = daily_counts.resample('ME').mean()
    ax5.plot(monthly_counts.index, monthly_counts.values, color='#2E86AB', linewidth=2)
    ax5.axhline(y=50, color='orange', linestyle='--', alpha=0.7)
    ax5.axvline(x=pd.Timestamp('2022-01-01'), color='red', linewidth=2, linestyle='--')
    ax5.fill_between(monthly_counts.index, 0, monthly_counts.values,
                     where=monthly_counts.index >= '2022-01-01', color='green', alpha=0.2)
    ax5.set_title('Symbol Count', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计摘要
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    yearly_stats = []
    for year in sorted(daily.index.year.unique()):
        yr = daily[daily.index.year == year]
        yr_ret = (1 + yr).prod() - 1
        yr_sharpe = yr.mean() / yr.std() * np.sqrt(365) if yr.std() > 0 else 0
        yearly_stats.append(f"{year}: {yr_ret:>6.0%}  (Sharpe: {yr_sharpe:.2f})")
    
    off_diag = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
    daily_n = len([k for k,v in FACTOR_CONFIG.items() if v.freq == 'daily'])
    intraday_n = len([k for k,v in FACTOR_CONFIG.items() if v.freq == 'intraday'])
    
    summary = f"""
        MULTI-FACTOR STRATEGY
        {'='*45}

        PERFORMANCE
        {'-'*45}
        Annual Return:     {result.annualized_return:>8.1%}
        Sharpe Ratio:      {result.sharpe_ratio:>8.2f}
        Max Drawdown:      {result.max_drawdown:>8.1%}
        Win Rate:          {result.win_rate:>8.1%}
        Final Value:       {result.total_net_value:>7.0f}x

        YEARLY
        {'-'*45}
        {chr(10).join(yearly_stats)}

        FACTORS
        {'-'*45}
        Daily:     {daily_n:>3}
        Intraday:  {intraday_n:>3}
        Max Corr:  {np.max(np.abs(off_diag)):.3f}
        """
    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_trading_costs(result: BacktestResult, benchmark_result: BacktestResult = None,
                       save_path: Optional[str] = None):
    """绘制交易成本分析图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Trading Cost Analysis', fontsize=14, fontweight='bold')
    
    # 1. 累计成本对比
    ax1 = axes[0]
    if result.daily_costs is not None:
        cum_cost = result.daily_costs.cumsum() * 100  # 转为百分比
        ax1.plot(cum_cost.index, cum_cost.values, color='#2E86AB', linewidth=2, label='Strategy')
        if benchmark_result and benchmark_result.daily_costs is not None:
            bm_cum_cost = benchmark_result.daily_costs.cumsum() * 100
            ax1.plot(bm_cum_cost.index, bm_cum_cost.values, color='#A23B72', 
                    linewidth=1.5, linestyle='--', label='Benchmark')
        ax1.set_title('Cumulative Trading Cost')
        ax1.set_ylabel('Cumulative Cost (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 每日成本分布
    ax2 = axes[1]
    if result.daily_costs is not None:
        daily_costs_bps = result.daily_costs * 10000  # 转换为bps
        ax2.hist(daily_costs_bps.dropna(), bins=50, alpha=0.7, color='#2E86AB', 
                edgecolor='white', label='Strategy')
        if benchmark_result and benchmark_result.daily_costs is not None:
            bm_daily_costs_bps = benchmark_result.daily_costs * 10000
            ax2.hist(bm_daily_costs_bps.dropna(), bins=50, alpha=0.5, color='#A23B72', 
                    edgecolor='white', label='Benchmark')
        ax2.axvline(x=daily_costs_bps.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {daily_costs_bps.mean():.1f} bps')
        ax2.set_title('Daily Cost Distribution')
        ax2.set_xlabel('Daily Cost (bps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 成本指标汇总表
    ax3 = axes[2]
    ax3.axis('off')
    
    metrics = [
        ['Metric', 'Strategy', 'Benchmark'],
        ['Total Cost', f'{result.total_cost:.2%}', 
         f'{benchmark_result.total_cost:.2%}' if benchmark_result else 'N/A'],
        ['Avg Daily Cost (bps)', f'{result.daily_costs.mean()*10000:.1f}' if result.daily_costs is not None else 'N/A',
         f'{benchmark_result.daily_costs.mean()*10000:.1f}' if benchmark_result and benchmark_result.daily_costs is not None else 'N/A'],
        ['Avg Daily Turnover', f'{result.avg_daily_turnover:.1%}',
         f'{benchmark_result.avg_daily_turnover:.1%}' if benchmark_result else 'N/A'],
        ['Avg Utilization', f'{result.avg_utilization:.1%}',
         f'{benchmark_result.avg_utilization:.1%}' if benchmark_result else 'N/A'],
        ['Sharpe Ratio', f'{result.sharpe_ratio:.2f}',
         f'{benchmark_result.sharpe_ratio:.2f}' if benchmark_result else 'N/A'],
        ['Ann. Return', f'{result.annualized_return:.1%}',
         f'{benchmark_result.annualized_return:.1%}' if benchmark_result else 'N/A'],
    ]
    
    table = ax3.table(cellText=metrics[1:], colLabels=metrics[0],
                      loc='center', cellLoc='center', colWidths=[0.4, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for j in range(3):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax3.set_title('Cost Metrics Summary', pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# VIP等级 × AUM 对比分析函数
# ============================================================

def run_vip_aum_backtests(combined: pd.DataFrame, backtester: Backtester,
                          long_pct: float = 0.2, short_pct: float = 0.2,
                          leverage: float = 1.0, impact_coef: float = 0.12,
                          max_participation: float = 0.05,
                          liquidity_filter: float = 0.4,
                          funding_kwargs: dict = None) -> dict:
    """
    运行所有VIP×AUM组合的回测
    
    Args:
        combined: 组合信号DataFrame
        backtester: Backtester实例
        long_pct: 做多比例
        short_pct: 做空比例
        leverage: 杠杆倍数
        impact_coef: 市场冲击系数
        max_participation: 最大参与率限制
    
    Returns:
        dict: 包含所有回测结果的字典
    """
    if funding_kwargs is None:
        funding_kwargs = {}

    print("\n" + "=" * 60)
    print("VIP × AUM COMBINATION BACKTEST")
    print("=" * 60)

    results = {}

    print("\n  Running Before Trading Cost baseline...")
    before_cost_result = backtester.run(
        combined, long_pct, short_pct,
        weight_method='sqrt_volume',
        leverage=leverage,
        enable_cost=False,
        aum=10_000_000,
        max_participation=max_participation,
        liquidity_filter=liquidity_filter,
        **funding_kwargs,
    )
    results['Before Trading Cost'] = {
        'sharpe': before_cost_result.sharpe_ratio,
        'ann_return': before_cost_result.annualized_return,
        'net_value': before_cost_result.total_net_value
    }
    
    total_combos = len(VIP_FEES) * len(AUM_LEVELS)
    combo_idx = 0
    
    for vip_name, taker_fee in VIP_FEES.items():
        for aum_name, aum_value in AUM_LEVELS.items():
            combo_idx += 1
            key = f"{vip_name}_{aum_name}"
            print(f"  [{combo_idx}/{total_combos}] {vip_name} @ {aum_name}...")
            
            result = backtester.run(
                combined, long_pct, short_pct,
                weight_method='sqrt_volume',
                leverage=leverage,
                enable_cost=True,
                taker_fee=taker_fee,
                impact_coef=impact_coef,
                aum=aum_value,
                max_participation=max_participation,
                liquidity_filter=liquidity_filter,
                **funding_kwargs
            )
            
            results[key] = {
                'sharpe': result.sharpe_ratio,
                'ann_return': result.annualized_return,
                'net_value': result.total_net_value,
                'total_cost': result.total_cost,
                'avg_turnover': result.avg_daily_turnover
            }
            
            print(f"       Sharpe: {result.sharpe_ratio:.2f}, Ann Return: {result.annualized_return:.1%}")
    
    return results


def plot_vip_aum_comparison(results: dict, save_path: Optional[str] = None):
    """
    创建VIP×AUM对比表格（合并显示年化收益和Sharpe）
    
    格式: 年化收益% (Sharpe)
    
    Args:
        results: run_vip_aum_backtests返回的结果字典
        save_path: 保存路径
    """
    vip_names = list(VIP_FEES.keys())
    aum_names = list(AUM_LEVELS.keys())
    fee_labels = {'VIP0': '5.0 bps', 'VIP6': '2.5 bps', 'VIP9': '1.7 bps'}
    
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('Strategy Performance by VIP Level × AUM\n(Binance USDT-M Futures)', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    ax.axis('off')
    
    # 构建合并表格，格式: 年化收益% (Sharpe)
    header = ['VIP Level', 'Taker Fee'] + [f'{aum}\nReturn (Sharpe)' for aum in aum_names]
    data = []
    
    # VIP行（从高费率到低费率：VIP0 → VIP6 → VIP9）
    for vip in vip_names:
        row = [vip, fee_labels[vip]]
        for aum in aum_names:
            key = f"{vip}_{aum}"
            if key in results:
                ann_ret = results[key]['ann_return']
                sharpe = results[key]['sharpe']
                row.append(f"{ann_ret:.1%} ({sharpe:.2f})")
            else:
                row.append('N/A')
        data.append(row)
    
    # Before Trading Cost放在最下面（数值放中间列$5M位置）
    if 'Before Trading Cost' in results:
        bc = results['Before Trading Cost']
        # 索引: 0=VIP Level, 1=Taker Fee, 2=$1M, 3=$5M(中间), 4=$10M
        data.append(['Before Trading Cost', '-', '', f"{bc['ann_return']:.1%} ({bc['sharpe']:.2f})", ''])
    
    table = ax.table(cellText=data, colLabels=header, loc='center', cellLoc='center',
                     colWidths=[0.14, 0.12] + [0.22] * len(aum_names))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    for j in range(len(header)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    for i in range(len(data)):
        row_idx = i + 1
        table[(row_idx, 0)].set_text_props(fontweight='bold')

    note_text = "Parameters: Long/Short 20% | Leverage 1.0x | Max Participation 5% | Impact Coef 0.12"
    fig.text(0.5, 0.08, note_text, ha='center', fontsize=9, 
             fontfamily='monospace', color='gray')
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.92])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close(fig)


def print_vip_aum_summary(results: dict):
    """打印VIP×AUM回测结果汇总"""
    print("\n" + "=" * 80)
    print("VIP × AUM RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Configuration':<25}{'Sharpe':>10}{'Ann.Return':>12}{'Net Value':>12}")
    print("-" * 60)
    
    if 'Before Trading Cost' in results:
        bc = results['Before Trading Cost']
        print(f"{'Before Trading Cost':<25}{bc['sharpe']:>10.2f}{bc['ann_return']:>11.1%}{bc['net_value']:>11.2f}x")
    
    print("-" * 60)
    
    for vip in VIP_FEES.keys():
        for aum in AUM_LEVELS.keys():
            key = f"{vip}_{aum}"
            if key in results:
                r = results[key]
                label = f"{vip} @ {aum}"
                print(f"{label:<25}{r['sharpe']:>10.2f}{r['ann_return']:>11.1%}{r['net_value']:>11.2f}x")


# ============================================================
# 主执行逻辑
# ============================================================

if __name__ == "__main__":
    
    # ==================== 配置 ====================
    DAILY_FACTORS_FILE = "./factors/daily_factors.csv"
    INTRADAY_FACTORS_FILE = "./factors/intraday_factors.csv"
    PRICE_DATA_DIR = "./futures_data"  # 日度价格数据目录（CSV格式）
    OUTPUT_DIR = "./output"
    TEST_START = "2022-01-01"
    LONG_PCT = 0.2
    SHORT_PCT = 0.2
    LEVERAGE = 1.0  # 杠杆倍数：1.0=多空各50%，2.0=多空各100%
    
    # ===== 样本划分 =====
    IN_SAMPLE_END = "2025-06-30"      # 样本内结束日期
    OUT_SAMPLE_START = "2025-07-01"   # 样本外开始日期
    OUT_SAMPLE_END = "2025-12-31"     # 样本外结束日期
    
    # 交易成本参数
    ENABLE_COST = True          # 是否启用交易成本
    TAKER_FEE = 0.0005          # Taker手续费 0.05%
    IMPACT_COEF = 0.12          # 市场冲击系数（订单簿实测校准，+20%安全边际）
    AUM = 1_000_000             # 策略资金规模（USDT）
    MAX_PARTICIPATION = 0.05   # 最大参与率限制（5%）
    
    # 流动性过滤参数
    LIQUIDITY_FILTER = 0.4     # 只做quote_volume前40%的标的（Q1+Q2）

    # 资金费率干预（信号调整 + 费率扣除，After/Before Trading Cost 一致生效）
    ENABLE_FUNDING = True        # 资金费率干预总开关
    COST_ADJ_PENALTY = 200       # 成本惩罚系数
    COST_ADJ_WINDOW = 14         # 预期费率滚动窗口（EWM span）
    COST_ADJ_METHOD = 'ewm'      # 预期费率计算方法（ewm/sma）



    # 是否运行VIP×AUM对比分析
    RUN_VIP_AUM_ANALYSIS = True
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("MULTI-FACTOR CRYPTO STRATEGY (Q1+Q2 Liquidity Filter)")
    print("=" * 60)
    
    # ==================== 加载因子 ====================
    print("\n[1/5] Loading factors from CSV files...")
    loader = FactorLoader(DAILY_FACTORS_FILE, INTRADAY_FACTORS_FILE)
    all_factors = loader.load_all_factors(normalize=True)
    print(f"Total factors: {len(all_factors)}")
    
    # ==================== 加载收益率 ====================
    print("\nLoading returns from price data...")
    fwd_ret, quote_volume, close_price, market_index = load_returns_from_price(PRICE_DATA_DIR)

    # 筛选测试期
    test_idx = fwd_ret.index[fwd_ret.index >= TEST_START]
    factors = {k: v.loc[v.index.intersection(test_idx)] for k, v in all_factors.items()}
    fwd_ret_test = fwd_ret.loc[test_idx]
    quote_volume_test = quote_volume.loc[quote_volume.index.intersection(test_idx)]
    close_price_test = close_price.loc[close_price.index.intersection(test_idx)]
    market_index_test = market_index.loc[market_index.index >= TEST_START]
    market_index_test = market_index_test / market_index_test.iloc[0]

    # ==================== 因子分析与回测 ====================
    print("\n[2/5] Running backtest...")

    analyzer = FactorAnalyzer(factors, fwd_ret_test)
    factor_stats = analyzer.analyze_all_factors()

    print("\nFactor Statistics:")
    print(factor_stats.to_string())
    
    # 等权组合信号
    common_dates = sorted(set.intersection(*[set(f.index) for f in factors.values()]))
    common_symbols = sorted(set.intersection(*[set(f.columns) for f in factors.values()]))
    combined = sum(
        FACTOR_CONFIG[name].direction * f.reindex(index=common_dates, columns=common_symbols).fillna(0)
        for name, f in factors.items()
    ) / len(factors)

    # 资金费率干预（信号调整 + 费率扣除）
    funding_kwargs = {}
    if ENABLE_FUNDING:
        print(f"\n  Applying funding cost adjustment (method={COST_ADJ_METHOD}, "
              f"penalty={COST_ADJ_PENALTY}, window={COST_ADJ_WINDOW}d)...")
        efr_wide = precompute_expected_funding_rate(
            funding_dir="./funding_rates",
            window=COST_ADJ_WINDOW,
            method=COST_ADJ_METHOD,
            verbose=True,
        )
        if not efr_wide.empty:
            efr_aligned = efr_wide.reindex(index=combined.index, columns=combined.columns).fillna(0)
            combined_raw = combined.copy()
            combined = combined - COST_ADJ_PENALTY * efr_aligned
            combined[combined_raw.isna()] = np.nan
            adj_nonzero = (efr_aligned != 0).sum().sum()
            total_slots = combined.shape[0] * combined.shape[1]
            print(f"  [CostAdj] Adjusted {adj_nonzero}/{total_slots} entries "
                  f"({adj_nonzero/total_slots*100:.1f}%)")

        print("\n  Loading funding rate costs...")
        fa = precompute_funding_costs(verbose=False)
        funding_kwargs = dict(funding_adj=fa)

    backtester = Backtester(fwd_ret_test, quote_volume_test, close_price_test)

    result = backtester.run(
        combined, LONG_PCT, SHORT_PCT,
        weight_method='sqrt_volume',
        leverage=LEVERAGE,
        enable_cost=ENABLE_COST,
        taker_fee=TAKER_FEE,
        impact_coef=IMPACT_COEF,
        aum=AUM,
        max_participation=MAX_PARTICIPATION,
        liquidity_filter=LIQUIDITY_FILTER,
        **funding_kwargs
    )

    # Benchmark回测 (before cost = 不含交易成本，资金费率干预与 after cost 一致)
    print("\n[3/5] Running benchmark (before cost)...")
    benchmark_result = backtester.run(
        combined, LONG_PCT, SHORT_PCT,
        weight_method='sqrt_volume',
        leverage=LEVERAGE,
        enable_cost=False,
        aum=AUM,
        max_participation=MAX_PARTICIPATION,
        liquidity_filter=LIQUIDITY_FILTER,
        **funding_kwargs,
    )
    
    # ==================== 打印结果 ====================
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS")
    print(f"Leverage: {LEVERAGE}x | Cost: {'ON' if ENABLE_COST else 'OFF'} | AUM: ${AUM/1e6:.0f}M | Liq.Filter: Top {LIQUIDITY_FILTER:.0%}")
    print("=" * 60)
    print(f"Period: {common_dates[0]} to {common_dates[-1]}")
    print(f"\n{'Metric':<25}{'After Trading Cost':>15}{'Before Trading Cost':>15}")
    print("-" * 55)
    print(f"{'Ann. Return':<25}{result.annualized_return:>14.1%}{benchmark_result.annualized_return:>14.1%}")
    print(f"{'Sharpe Ratio':<25}{result.sharpe_ratio:>15.2f}{benchmark_result.sharpe_ratio:>15.2f}")
    print(f"{'Max Drawdown':<25}{result.max_drawdown:>14.1%}{benchmark_result.max_drawdown:>14.1%}")
    print(f"{'Net Value':<25}{result.total_net_value:>14.2f}x{benchmark_result.total_net_value:>13.2f}x")
    
    if ENABLE_COST:
        print(f"\n{'--- Cost Analysis ---':<25}")
        print(f"{'Total Cost':<25}{result.total_cost:>14.1%}")
        print(f"{'Avg Daily Turnover':<25}{result.avg_daily_turnover:>14.1%}")
        print(f"{'Cost/Return Ratio':<25}{result.cost_return_ratio:>14.1%}")
        print(f"{'Avg Utilization':<25}{result.avg_utilization:>14.1%}")
    
    # ==================== 生成图表 ====================
    print("\n[4/5] Generating charts...")

    plot_backtest_results(
        result,
        benchmark_result=benchmark_result,
        market_index=market_index_test,
        title=f"Strategy Performance (Sharpe: {result.sharpe_ratio:.2f}, Leverage: {LEVERAGE}x)",
        save_path=f"{OUTPUT_DIR}/strategy_performance.png",
        show_cost_metrics=ENABLE_COST
    )

    plot_monthly_returns(
        result.daily_returns,
        save_path=f"{OUTPUT_DIR}/monthly_returns.png"
    )
    
    # ==================== 样本外回测 ====================
    print("\n[4.5/5] Running out-of-sample backtest...")
    print(f"  Period: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END}")
    
    oos_idx = fwd_ret_test.index[(fwd_ret_test.index >= OUT_SAMPLE_START) & 
                                  (fwd_ret_test.index <= OUT_SAMPLE_END)]
    
    if len(oos_idx) > 0:
        # 样本外的因子和数据
        oos_common_dates = [d for d in common_dates if OUT_SAMPLE_START <= str(d)[:10] <= OUT_SAMPLE_END]
        
        if len(oos_common_dates) > 0:
            combined_oos = combined.loc[combined.index.isin(oos_common_dates)]
            fwd_ret_oos = fwd_ret_test.loc[fwd_ret_test.index.isin(oos_common_dates)]
            qv_oos = quote_volume_test.loc[quote_volume_test.index.isin(oos_common_dates)]
            cp_oos = close_price_test.loc[close_price_test.index.isin(oos_common_dates)]
            market_index_oos = market_index_test.loc[market_index_test.index.isin(oos_common_dates)]
            if len(market_index_oos) > 0:
                market_index_oos = market_index_oos / market_index_oos.iloc[0]
            
            oos_backtester = Backtester(fwd_ret_oos, qv_oos, cp_oos)

            oos_result = oos_backtester.run(
                combined_oos, LONG_PCT, SHORT_PCT,
                weight_method='sqrt_volume',
                leverage=LEVERAGE,
                enable_cost=ENABLE_COST,
                taker_fee=TAKER_FEE,
                impact_coef=IMPACT_COEF,
                aum=AUM,
                max_participation=MAX_PARTICIPATION,
                liquidity_filter=LIQUIDITY_FILTER,
                **funding_kwargs
            )

            # 样本外回测 (before cost = 不含交易成本)
            oos_benchmark = oos_backtester.run(
                combined_oos, LONG_PCT, SHORT_PCT,
                weight_method='sqrt_volume',
                leverage=LEVERAGE,
                enable_cost=False,
                aum=AUM,
                max_participation=MAX_PARTICIPATION,
                liquidity_filter=LIQUIDITY_FILTER,
                **funding_kwargs,
            )
            
            print(f"\n  Out-of-Sample Results ({len(oos_common_dates)} days):")
            print(f"    Sharpe (after cost):  {oos_result.sharpe_ratio:.2f}")
            print(f"    Sharpe (before cost): {oos_benchmark.sharpe_ratio:.2f}")
            print(f"    Annual Return:        {oos_result.annualized_return:.1%}")
            print(f"    Max Drawdown:         {oos_result.max_drawdown:.1%}")
            print(f"    Net Value:            {oos_result.total_net_value:.2f}x")
            
            plot_out_of_sample_results(
                oos_result, 
                benchmark_result=oos_benchmark,
                market_index=market_index_oos if len(market_index_oos) > 0 else None,
                title=f"Out-of-Sample Performance (Sharpe: {oos_result.sharpe_ratio:.2f})",
                save_path=f"{OUTPUT_DIR}/out_of_sample_performance.png",
                show_cost_metrics=ENABLE_COST,
                oos_start=OUT_SAMPLE_START,
                oos_end=OUT_SAMPLE_END
            )
        else:
            print("  Warning: No data available for out-of-sample period!")
    else:
        print("  Warning: Out-of-sample period has no data!")
    
    # ==================== VIP × AUM 对比分析 ====================
    if RUN_VIP_AUM_ANALYSIS:
        print("\n[5/5] Running VIP × AUM comparison analysis...")
        
        vip_aum_results = run_vip_aum_backtests(
            combined, backtester,
            long_pct=LONG_PCT,
            short_pct=SHORT_PCT,
            leverage=LEVERAGE,
            impact_coef=IMPACT_COEF,
            max_participation=MAX_PARTICIPATION,
            liquidity_filter=LIQUIDITY_FILTER,
            funding_kwargs=funding_kwargs
        )
        
        print_vip_aum_summary(vip_aum_results)

        plot_vip_aum_comparison(
            vip_aum_results,
            save_path=f"{OUTPUT_DIR}/vip_aum_comparison.png"
        )
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print(f"Charts saved to: {OUTPUT_DIR}/")
    print("=" * 60)