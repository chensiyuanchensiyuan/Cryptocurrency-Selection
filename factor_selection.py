"""
因子筛选优化脚本（带交易成本 + 样本外测试）

目标：从合格因子池中选择最优因子组合
约束：因子间相关性 <= 0.7
优化目标：最大化扣除交易成本后的组合Sharpe

方法：
1. 贪心算法（快速）
2. 启发式搜索（随机扰动优化）

样本划分：
- 样本内 (In-Sample): 2022-01-01 ~ 2025-06-30 (用于因子筛选和优化)
- 样本外 (Out-of-Sample): 2025-07-01 ~ 2025-12-31 (用于验证)

交易成本模型：
- Taker手续费: 0.05%
- 市场冲击: c × σ × √(participation_rate)，c=0.12
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from crypto_strategy import (Backtester, BacktestResult, load_returns_from_price,
                             precompute_funding_costs, zscore, FACTOR_CONFIG)


# ============================================================
# 配置
# ============================================================

FACTOR_DIR = "./factors"
DAILY_DATA_DIR = "./futures_data"
FUNDING_DIR = "./funding_rates"
QUALIFIED_FACTORS_FILE = "./screening_results/qualified_factors.csv"

# ===== 样本划分 =====
IN_SAMPLE_START = "2022-01-01"
IN_SAMPLE_END = "2025-06-30"
OUT_SAMPLE_START = "2025-07-01"
OUT_SAMPLE_END = "2025-12-31"

# 因子筛选约束
MAX_CORR = 0.7        # 最大允许相关性
MIN_FACTORS = 8       # 最少因子数量

# 因子数量奖励（体现分散化带来的稳定性）
# 逻辑：因子越多，单个因子失效对总信号影响越小
# 例如：FACTOR_BONUS=0.03 意味着每多1个因子，允许Sharpe下降0.03
FACTOR_BONUS = 0.03

# 初始因子组合（上一轮优化结果，作为搜索起点）
INITIAL_FACTORS = [
    'trade_size_skew',
    'max_buy_pressure_1h',
    'market_r2',
    'early_main_diff',
    'info_entropy',
    'amihud_illiq_log',
    'lower_shadow',
    'max_drawdown_intra',
]

# 搜索参数
N_ITERATIONS = 500  # 迭代次数

# 交易成本参数（与主策略一致）
ENABLE_COST = True
TAKER_FEE = 0.0005        # Taker手续费 0.05%
IMPACT_COEF = 0.12        # 市场冲击系数（订单簿实测校准，+20%安全边际）
AUM = 1_000_000           # 策略资金规模（USDT）
MAX_PARTICIPATION = 0.05  # 最大参与率限制（5%）
LEVERAGE = 1.0            # 杠杆倍数
LIQUIDITY_FILTER = 0.4    # 流动性过滤：只做quote_volume前40%的标的

# 资金费率参数
ENABLE_FUNDING_COST = True  # 是否启用资金费率成本


# ============================================================
# 数据加载
# ============================================================

def load_factors() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载日频和日内因子"""
    daily = pd.read_csv(f'{FACTOR_DIR}/daily_factors.csv')
    daily['date'] = pd.to_datetime(daily['date'])
    
    intraday = pd.read_csv(f'{FACTOR_DIR}/intraday_factors.csv')
    intraday['date'] = pd.to_datetime(intraday['date'])
    
    return daily, intraday


def load_price_data():
    """加载价格数据并计算前向收益率"""
    fwd_ret, quote_volume, close_price, _ = load_returns_from_price(DAILY_DATA_DIR)
    return fwd_ret, quote_volume, close_price


def build_funding_kwargs() -> dict:
    """预计算资金费率数据，返回可直接传给 Backtester.run() 的 kwargs"""
    if not ENABLE_FUNDING_COST:
        return {}
    fa = precompute_funding_costs(verbose=True)
    return dict(funding_adj=fa)


def load_qualified_factors() -> pd.DataFrame:
    """加载合格因子列表（已由 factor_screening.py 筛选和去重）"""
    df = pd.read_csv(QUALIFIED_FACTORS_FILE)
    return df.sort_values('sharpe', ascending=False).reset_index(drop=True)


# ============================================================
# 因子处理
# ============================================================


def get_factor_wide(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, 
                    factor_name: str) -> pd.DataFrame:
    """获取因子宽表"""
    if factor_name in daily_df.columns:
        df = daily_df[['date', 'symbol', factor_name]].copy()
    elif factor_name in intraday_df.columns:
        df = intraday_df[['date', 'symbol', factor_name]].copy()
    else:
        return None

    wide = df.pivot(index='date', columns='symbol', values=factor_name)
    return zscore(wide)


def calc_factor_ic(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """计算因子IC序列"""
    idx = factor.index.intersection(returns.index)
    cols = factor.columns.intersection(returns.columns)

    ic_list = []
    for d in idx:
        f_val = factor.loc[d, cols].values
        r_val = returns.loc[d, cols].values
        valid = ~(np.isnan(f_val) | np.isnan(r_val))
        if valid.sum() >= 20:
            ic, _ = stats.spearmanr(f_val[valid], r_val[valid])
            ic_list.append((d, ic))

    return pd.Series(dict(ic_list))


def calc_signal_stability(signals: pd.DataFrame) -> Dict[str, float]:
    """
    计算信号稳定性指标

    Args:
        signals: 信号矩阵 (date x symbol)

    Returns:
        包含多个稳定性指标的字典:
        - signal_autocorr: 信号日间相关性（越高越稳定）
        - signal_persistence: 信号符号持续率（越高越稳定）
        - rank_stability: 排名稳定性（越高越稳定）
    """
    if len(signals) < 2:
        return {'signal_autocorr': 0, 'signal_persistence': 0, 'rank_stability': 0}

    # 1. 信号日间相关性：连续两天信号的截面相关性
    autocorrs = []
    for i in range(1, len(signals)):
        s_prev = signals.iloc[i-1].dropna()
        s_curr = signals.iloc[i].dropna()
        common = s_prev.index.intersection(s_curr.index)
        if len(common) >= 20:
            corr = s_prev[common].corr(s_curr[common])
            if not np.isnan(corr):
                autocorrs.append(corr)
    signal_autocorr = np.mean(autocorrs) if autocorrs else 0

    # 2. 信号符号持续率：信号方向不变的比例
    sign_changes = []
    for i in range(1, len(signals)):
        s_prev = np.sign(signals.iloc[i-1])
        s_curr = np.sign(signals.iloc[i])
        common = s_prev.dropna().index.intersection(s_curr.dropna().index)
        if len(common) >= 20:
            same_sign = (s_prev[common] == s_curr[common]).mean()
            sign_changes.append(same_sign)
    signal_persistence = np.mean(sign_changes) if sign_changes else 0

    # 3. 排名稳定性：连续两天排名的相关性
    rank_corrs = []
    for i in range(1, len(signals)):
        r_prev = signals.iloc[i-1].rank()
        r_curr = signals.iloc[i].rank()
        common = r_prev.dropna().index.intersection(r_curr.dropna().index)
        if len(common) >= 20:
            corr = r_prev[common].corr(r_curr[common])
            if not np.isnan(corr):
                rank_corrs.append(corr)
    rank_stability = np.mean(rank_corrs) if rank_corrs else 0

    return {
        'signal_autocorr': signal_autocorr,
        'signal_persistence': signal_persistence,
        'rank_stability': rank_stability
    }


def calc_adjusted_score(sharpe: float, n_factors: int,
                        min_factors: int = MIN_FACTORS,
                        factor_bonus: float = FACTOR_BONUS) -> float:
    """
    计算调整后的优化得分

    公式: adjusted_score = sharpe + factor_bonus * (n_factors - min_factors)

    逻辑：因子越多，单个因子失效对总信号影响越小（分散化稳定性）
    - 8个因子：单因子失效影响 12.5%
    - 10个因子：单因子失效影响 10%
    - 12个因子：单因子失效影响 8.3%

    Args:
        sharpe: 策略夏普比率
        n_factors: 因子数量
        min_factors: 最小因子数
        factor_bonus: 每多一个因子的奖励（允许Sharpe下降的额度）

    Returns:
        调整后得分
    """
    factor_reward = factor_bonus * (n_factors - min_factors)
    return sharpe + factor_reward


# ============================================================
# 回测函数（带交易成本）
# ============================================================

def backtest_combination(factors: Dict[str, pd.DataFrame],
                         returns: pd.DataFrame,
                         quote_volume: pd.DataFrame,
                         close_price: pd.DataFrame = None,
                         funding_kwargs: dict = None,
                         enable_cost: bool = True,
                         start_date: str = None,
                         end_date: str = None,
                         calc_stability: bool = False) -> Dict:
    """
    回测因子组合（使用主策略的带交易成本回测器）

    Args:
        factors: 因子宽表字典
        returns: 收益率
        quote_volume: 成交额
        close_price: 收盘价（用于波动率计算）
        funding_kwargs: 资金费率参数字典，直接传给 Backtester.run()
        enable_cost: 是否启用交易成本（手续费+冲击）
        start_date: 回测起始日期（可选）
        end_date: 回测结束日期（可选）
        calc_stability: 是否计算信号稳定性指标

    Returns:
        包含sharpe, ann_ret, max_dd, net_value等的字典
    """
    if funding_kwargs is None:
        funding_kwargs = {}

    common_dates = sorted(set.intersection(*[set(f.index) for f in factors.values()]))
    common_symbols = sorted(set.intersection(*[set(f.columns) for f in factors.values()]))

    if start_date:
        common_dates = [d for d in common_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        common_dates = [d for d in common_dates if d <= pd.Timestamp(end_date)]

    if len(common_dates) < 30 or len(common_symbols) < 20:
        return {
            'sharpe': 0, 'ann_ret': 0, 'max_dd': 1, 'net_value': 1,
            'total_cost': 0, 'sharpe_before': 0, 'avg_turnover': 0,
            'n_days': len(common_dates),
            'signal_autocorr': 0, 'signal_persistence': 0, 'rank_stability': 0,
            'adjusted_score': 0
        }

    combined = sum(
        (FACTOR_CONFIG[name].direction if name in FACTOR_CONFIG else 1) *
        f.reindex(index=common_dates, columns=common_symbols).fillna(0)
        for name, f in factors.items()
    ) / len(factors)

    if calc_stability:
        stability_metrics = calc_signal_stability(combined)
    else:
        stability_metrics = {'signal_autocorr': 0, 'signal_persistence': 0, 'rank_stability': 0}

    fwd = returns.reindex(index=common_dates, columns=common_symbols)
    qv = quote_volume.reindex(index=common_dates, columns=common_symbols)
    cp = None
    if close_price is not None:
        cp = close_price.reindex(index=common_dates, columns=common_symbols)

    backtester = Backtester(fwd, qv, cp)
    result = backtester.run(
        signals=combined,
        long_pct=0.2,
        short_pct=0.2,
        weight_method='sqrt_volume',
        leverage=LEVERAGE,
        enable_cost=enable_cost,
        taker_fee=TAKER_FEE,
        impact_coef=IMPACT_COEF,
        aum=AUM,
        max_participation=MAX_PARTICIPATION,
        liquidity_filter=LIQUIDITY_FILTER,
        **funding_kwargs
    )

    adjusted_score = calc_adjusted_score(
        sharpe=result.sharpe_ratio,
        n_factors=len(factors)
    )

    return {
        'sharpe': result.sharpe_ratio,
        'ann_ret': result.annualized_return,
        'max_dd': result.max_drawdown,
        'net_value': result.total_net_value,
        'total_cost': result.total_cost,
        'sharpe_before': result.sharpe_before_cost,
        'avg_turnover': result.avg_daily_turnover,
        'n_days': len(common_dates),
        'cumulative_returns': result.cumulative_returns,
        'daily_returns': result.daily_returns,
        'signal_autocorr': stability_metrics['signal_autocorr'],
        'signal_persistence': stability_metrics['signal_persistence'],
        'rank_stability': stability_metrics['rank_stability'],
        'adjusted_score': adjusted_score
    }


# ============================================================
# 因子筛选算法
# ============================================================

def is_valid_combination(combo: List[str], corr_matrix: pd.DataFrame, max_corr: float) -> bool:
    """检查组合是否满足相关性约束"""
    for i, f1 in enumerate(combo):
        for f2 in combo[i+1:]:
            if abs(corr_matrix.loc[f1, f2]) > max_corr:
                return False
    return True


def greedy_selection(factor_sharpe: Dict[str, float], 
                     corr_matrix: pd.DataFrame,
                     max_corr: float = 0.7,
                     max_factors: int = 15) -> List[str]:
    """贪心算法：按Sharpe降序选择，跳过高相关因子"""
    sorted_factors = sorted(factor_sharpe.keys(), 
                           key=lambda x: factor_sharpe.get(x, 0), 
                           reverse=True)
    sorted_factors = [f for f in sorted_factors if f in corr_matrix.columns]
    
    selected = []
    for factor in sorted_factors:
        if len(selected) >= max_factors:
            break
        if all(abs(corr_matrix.loc[factor, s]) <= max_corr for s in selected):
            selected.append(factor)
    
    return selected


def smart_search(factor_names: List[str],
                 factor_sharpe: Dict[str, float],
                 corr_matrix: pd.DataFrame,
                 factors_wide: Dict[str, pd.DataFrame],
                 returns: pd.DataFrame,
                 quote_volume: pd.DataFrame,
                 close_price: pd.DataFrame,
                 funding_kwargs: dict = None,
                 max_corr: float = 0.7,
                 min_factors: int = 8,
                 n_iterations: int = 200,
                 initial_combo: List[str] = None,
                 start_date: str = None,
                 end_date: str = None) -> Tuple[List[str], float, Dict]:
    """
    启发式搜索：从初始解出发，随机扰动寻找更优解

    优化目标：带交易成本（含资金费率）的Sharpe
    """
    
    available = [f for f in factor_names if f in corr_matrix.columns]
    
    # 初始解：使用自定义组合或贪心算法
    if initial_combo is not None:
        # 过滤掉不存在的因子
        current_combo = [f for f in initial_combo if f in available]
        print(f"  Using custom initial combo: {len(current_combo)} factors")
    else:
        current_combo = greedy_selection(factor_sharpe, corr_matrix, max_corr)
        print(f"  Using greedy initial combo: {len(current_combo)} factors")
    
    # 确保满足最小因子数量
    while len(current_combo) < min_factors:
        candidates = [f for f in available if f not in current_combo]
        if not candidates:
            break
        best_candidate = None
        best_max_corr = 1.0
        for c in candidates:
            max_c = max(abs(corr_matrix.loc[c, s]) for s in current_combo) if current_combo else 0
            if max_c < best_max_corr:
                best_max_corr = max_c
                best_candidate = c
        if best_candidate:
            current_combo.append(best_candidate)
        else:
            break
    
    # 评估初始解（带交易成本+资金费率）
    combo_factors = {f: factors_wide[f] for f in current_combo}
    current_result = backtest_combination(
        combo_factors, returns, quote_volume, close_price,
        funding_kwargs=funding_kwargs,
        enable_cost=ENABLE_COST,
        start_date=start_date, end_date=end_date
    )
    current_score = current_result['adjusted_score']

    best_combo = current_combo.copy()
    best_score = current_score
    best_result = current_result

    cost_label = f"cost={'on' if ENABLE_COST else 'off'}, funding={'on' if ENABLE_FUNDING_COST else 'off'}"
    print(f"  Initial: {len(current_combo)} factors, Sharpe={current_result['sharpe']:.3f}, "
          f"AdjScore={current_score:.3f} ({cost_label})")
    
    no_improve_count = 0  # 连续无改进计数
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(n_iterations), desc="  Optimizing", ncols=80, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for i in pbar:
        # 随机操作选择
        # 如果连续多次无改进，增加大扰动概率
        if no_improve_count > 20:
            op = np.random.choice(['add', 'remove', 'replace', 'replace2', 'replace3', 'shuffle'], 
                                  p=[0.1, 0.1, 0.2, 0.25, 0.25, 0.1])
        else:
            op = np.random.choice(['add', 'remove', 'replace', 'replace2'], 
                                  p=[0.25, 0.2, 0.35, 0.2])
        
        new_combo = current_combo.copy()
        
        if op == 'add' and len(new_combo) < 15:
            candidates = [f for f in available if f not in new_combo]
            if candidates:
                new_combo.append(np.random.choice(candidates))
        
        elif op == 'remove' and len(new_combo) > min_factors:
            new_combo.remove(np.random.choice(new_combo))
        
        elif op == 'replace' and len(new_combo) > 0:
            to_remove = np.random.choice(new_combo)
            candidates = [f for f in available if f not in new_combo]
            if candidates:
                new_combo.remove(to_remove)
                new_combo.append(np.random.choice(candidates))
        
        elif op == 'replace2' and len(new_combo) >= 2:
            # 大扰动：一次替换2个因子
            to_remove = np.random.choice(new_combo, size=min(2, len(new_combo)), replace=False).tolist()
            for f in to_remove:
                new_combo.remove(f)
            candidates = [f for f in available if f not in new_combo]
            if len(candidates) >= 2:
                to_add = np.random.choice(candidates, size=2, replace=False).tolist()
                new_combo.extend(to_add)
        
        elif op == 'replace3' and len(new_combo) >= 3:
            # 更大扰动：一次替换3个因子
            to_remove = np.random.choice(new_combo, size=min(3, len(new_combo)), replace=False).tolist()
            for f in to_remove:
                new_combo.remove(f)
            candidates = [f for f in available if f not in new_combo]
            if len(candidates) >= 3:
                to_add = np.random.choice(candidates, size=3, replace=False).tolist()
                new_combo.extend(to_add)
        
        elif op == 'shuffle':
            # 随机重启：保留最好的3个因子，其余随机选择
            top_factors = sorted(new_combo, key=lambda x: factor_sharpe.get(x, 0), reverse=True)[:3]
            candidates = [f for f in available if f not in top_factors]
            n_new = min(min_factors - 3, len(candidates))
            if n_new > 0:
                new_factors = np.random.choice(candidates, size=n_new, replace=False).tolist()
                new_combo = top_factors + new_factors
        
        # 检查约束
        if len(new_combo) < min_factors:
            continue
        if not is_valid_combination(new_combo, corr_matrix, max_corr):
            continue
        
        # 评估新解（带交易成本+资金费率）
        combo_factors = {f: factors_wide[f] for f in new_combo}
        result = backtest_combination(
            combo_factors, returns, quote_volume, close_price,
            funding_kwargs=funding_kwargs,
            enable_cost=ENABLE_COST,
            start_date=start_date, end_date=end_date
        )
        new_score = result['adjusted_score']

        # 模拟退火：更高的初始温度，更慢的降温
        accept = False
        if new_score > current_score:
            accept = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            # 温度：从0.2开始递减到0.01
            temp = 0.2 * (1 - i / n_iterations) + 0.01
            delta = new_score - current_score
            if np.random.random() < np.exp(delta / temp):
                accept = True

        if accept:
            current_combo = new_combo
            current_score = new_score

            if new_score > best_score:
                best_combo = new_combo.copy()
                best_score = new_score
                best_result = result
                pbar.set_postfix({'score': f'{best_score:.3f}', 'factors': len(best_combo)})
                tqdm.write(f"  Iter {i+1}: {len(best_combo)} factors, Sharpe={result['sharpe']:.3f}, "
                          f"AdjScore={best_score:.3f} [op={op}]")
        
        # 每50次迭代，如果还没改进，尝试从best重新开始
        if (i + 1) % 50 == 0 and no_improve_count > 30:
            current_combo = best_combo.copy()
            current_score = best_score
            no_improve_count = 0
            tqdm.write(f"  Iter {i+1}: Restart from best solution")

    return best_combo, best_score, best_result


def run_out_of_sample_test(best_combo: List[str],
                           factors_wide: Dict[str, pd.DataFrame],
                           returns: pd.DataFrame,
                           quote_volume: pd.DataFrame,
                           close_price: pd.DataFrame,
                           funding_kwargs: dict = None,
                           start_date: str = None,
                           end_date: str = None) -> Dict:
    """
    对选定的因子组合在样本外期间进行回测

    Returns:
        {'with_cost': result_dict, 'no_cost': result_dict}
    """
    combo_factors = {f: factors_wide[f] for f in best_combo}

    # 带成本回测（含资金费率）
    result_with_cost = backtest_combination(
        combo_factors, returns, quote_volume, close_price,
        funding_kwargs=funding_kwargs,
        enable_cost=True,
        start_date=start_date, end_date=end_date
    )

    # 不带成本回测（用于对比）
    result_no_cost = backtest_combination(
        combo_factors, returns, quote_volume, close_price,
        funding_kwargs=None,
        enable_cost=False,
        start_date=start_date, end_date=end_date
    )

    return {
        'with_cost': result_with_cost,
        'no_cost': result_no_cost
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("FACTOR SELECTION OPTIMIZATION (WITH OUT-OF-SAMPLE TEST)")
    print("=" * 70)
    print(f"\nSample Split:")
    print(f"  - In-Sample:     {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
    print(f"  - Out-of-Sample: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END}")
    print(f"\nConstraints:")
    print(f"  - Max correlation: {MAX_CORR}")
    print(f"  - Min factors: {MIN_FACTORS}")
    print(f"\nDiversification Bonus:")
    print(f"  - Factor bonus: {FACTOR_BONUS} (Sharpe tolerance per extra factor)")
    print(f"  - Formula: AdjScore = Sharpe + {FACTOR_BONUS}×(n_factors - {MIN_FACTORS})")
    print(f"\nTrading Cost Model:")
    print(f"  - Enable cost: {ENABLE_COST}")
    print(f"  - Taker fee: {TAKER_FEE:.2%}")
    print(f"  - Impact coef: {IMPACT_COEF}")
    print(f"  - AUM: ${AUM/1e6:.0f}M")
    print(f"  - Max participation: {MAX_PARTICIPATION:.0%}")
    print(f"  - Liquidity filter: Top {LIQUIDITY_FILTER:.0%}")
    print(f"\nFunding Rate Cost:")
    print(f"  - Enable funding cost: {ENABLE_FUNDING_COST}")
    print(f"\nInitial Factors ({len(INITIAL_FACTORS)}):")
    for f in INITIAL_FACTORS:
        print(f"  - {f}")
    print(f"\nObjective: maximize Sharpe ratio {'after cost' if ENABLE_COST else 'before cost'}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/6] Loading data...")
    daily_factors, intraday_factors = load_factors()
    returns, quote_volume, close_price = load_price_data()
    funding_kwargs = build_funding_kwargs()
    qualified_df = load_qualified_factors()
    
    print(f"  Qualified factors: {len(qualified_df)}")
    for _, row in qualified_df.iterrows():
        print(f"    {row['factor']:<25} Sharpe={row['sharpe']:.2f}")
    
    # 2. 获取因子宽表（全量数据，后续按日期切分）
    print("\n[2/6] Processing factors...")
    factors_wide = {}
    factor_sharpe = {}
    
    for _, row in qualified_df.iterrows():
        fname = row['factor']
        wide = get_factor_wide(daily_factors, intraday_factors, fname)
        if wide is not None:
            # 只保留2022年之后的数据
            wide = wide[wide.index >= IN_SAMPLE_START]
            factors_wide[fname] = wide
            factor_sharpe[fname] = row['sharpe']
    
    print(f"  Loaded {len(factors_wide)} factors")
    
    # 3. 计算IC相关性矩阵（仅使用样本内数据）
    print("\n[3/6] Computing IC correlation matrix (in-sample only)...")
    
    # 筛选样本内收益率
    returns_in_sample = returns[(returns.index >= IN_SAMPLE_START) & 
                                 (returns.index <= IN_SAMPLE_END)]
    
    ic_series = {}
    for fname, fwide in factors_wide.items():
        # 筛选样本内因子
        fwide_in = fwide[(fwide.index >= IN_SAMPLE_START) & 
                         (fwide.index <= IN_SAMPLE_END)]
        ic = calc_factor_ic(fwide_in, returns_in_sample)
        if len(ic) > 0:
            ic_series[fname] = ic
    
    ic_df = pd.DataFrame(ic_series)
    corr_matrix = ic_df.corr()
    
    # 显示高相关对
    print(f"\n  High correlation pairs (|corr| > {MAX_CORR}):")
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > MAX_CORR:
                high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], c))
                print(f"    {corr_matrix.index[i]:<25} <-> {corr_matrix.columns[j]:<25}: {c:.2f}")
    
    if not high_corr_pairs:
        print("    (None)")
    
    # 4. 因子筛选（仅使用样本内数据）
    print("\n[4/6] Running optimization (in-sample)...")
    print(f"  Initial factors: {INITIAL_FACTORS}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  Period: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
    
    # 筛选样本内数据
    qv_in_sample = quote_volume[(quote_volume.index >= IN_SAMPLE_START) &
                                 (quote_volume.index <= IN_SAMPLE_END)]
    cp_in_sample = close_price[(close_price.index >= IN_SAMPLE_START) &
                                (close_price.index <= IN_SAMPLE_END)]

    best_combo, best_score, best_result = smart_search(
        list(factors_wide.keys()), factor_sharpe, corr_matrix,
        factors_wide, returns_in_sample, qv_in_sample, cp_in_sample,
        funding_kwargs=funding_kwargs,
        max_corr=MAX_CORR, min_factors=MIN_FACTORS,
        n_iterations=N_ITERATIONS, initial_combo=INITIAL_FACTORS,
        start_date=IN_SAMPLE_START, end_date=IN_SAMPLE_END
    )

    # 5. 样本外测试
    print("\n[5/6] Running out-of-sample test...")
    print(f"  Period: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END}")
    print(f"  Selected factors: {best_combo}")

    oos_results = run_out_of_sample_test(
        best_combo, factors_wide, returns, quote_volume, close_price,
        funding_kwargs=funding_kwargs,
        start_date=OUT_SAMPLE_START, end_date=OUT_SAMPLE_END
    )
    
    # 6. 输出结果
    print("\n" + "=" * 70)
    print("OPTIMAL FACTOR COMBINATION")
    print("=" * 70)
    
    print(f"\nNumber of factors: {len(best_combo)}")
    print(f"\nSelected factors (sorted by single-factor Sharpe):")
    sorted_best = sorted(best_combo, key=lambda x: factor_sharpe[x], reverse=True)
    for i, f in enumerate(sorted_best):
        print(f"  {i+1}. {f:<25} Sharpe={factor_sharpe[f]:.2f}")
    
    # 样本内结果
    print(f"\n{'='*70}")
    print("IN-SAMPLE RESULTS")
    print(f"Period: {IN_SAMPLE_START} ~ {IN_SAMPLE_END} ({best_result['n_days']} days)")
    print(f"{'='*70}")
    print(f"  Sharpe (after cost):  {best_result['sharpe']:.2f}")
    print(f"  Sharpe (before cost): {best_result['sharpe_before']:.2f}")
    print(f"  Annual Return:        {best_result['ann_ret']:.1%}")
    print(f"  Max Drawdown:         {best_result['max_dd']:.1%}")
    print(f"  Net Value:            {best_result['net_value']:.2f}x")
    print(f"\n  Cost Breakdown:")
    print(f"    Trading Cost:       {best_result['total_cost']:.2%}")
    print(f"  Avg Daily Turnover:   {best_result['avg_turnover']:.1%}")
    print(f"\n  Diversification Metrics:")
    print(f"    Number of factors:     {len(best_combo)}")
    print(f"    Single-factor impact:  {100/len(best_combo):.1f}%")
    print(f"  Adjusted Score:          {best_result['adjusted_score']:.3f}")
    print(f"    = Sharpe({best_result['sharpe']:.2f}) + {FACTOR_BONUS:.2f}×({len(best_combo)}-{MIN_FACTORS})")
    
    # 样本外结果
    oos_with_cost = oos_results['with_cost']
    oos_no_cost = oos_results['no_cost']
    
    print(f"\n{'='*70}")
    print("OUT-OF-SAMPLE RESULTS")
    print(f"Period: {OUT_SAMPLE_START} ~ {OUT_SAMPLE_END} ({oos_with_cost['n_days']} days)")
    print(f"{'='*70}")
    print(f"  Sharpe (after cost):  {oos_with_cost['sharpe']:.2f}")
    print(f"  Sharpe (before cost): {oos_no_cost['sharpe']:.2f}")
    print(f"  Annual Return:        {oos_with_cost['ann_ret']:.1%}")
    print(f"  Max Drawdown:         {oos_with_cost['max_dd']:.1%}")
    print(f"  Net Value:            {oos_with_cost['net_value']:.2f}x")
    print(f"\n  Cost Breakdown:")
    print(f"    Trading Cost:       {oos_with_cost['total_cost']:.2%}")
    print(f"  Avg Daily Turnover:   {oos_with_cost['avg_turnover']:.1%}")
    print(f"\n  Diversification Metrics:")
    print(f"    Number of factors:     {len(best_combo)}")
    print(f"    Single-factor impact:  {100/len(best_combo):.1f}%")
    
    # 样本内外对比
    print(f"\n{'='*70}")
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25}{'In-Sample':>15}{'Out-of-Sample':>15}{'Diff':>12}")
    print("-" * 67)

    sharpe_diff = oos_with_cost['sharpe'] - best_result['sharpe']
    ann_ret_diff = oos_with_cost['ann_ret'] - best_result['ann_ret']

    print(f"{'Sharpe (after cost)':<25}{best_result['sharpe']:>15.2f}{oos_with_cost['sharpe']:>15.2f}{sharpe_diff:>+12.2f}")
    print(f"{'Annual Return':<25}{best_result['ann_ret']:>14.1%}{oos_with_cost['ann_ret']:>14.1%}{ann_ret_diff:>+11.1%}")
    print(f"{'Max Drawdown':<25}{best_result['max_dd']:>14.1%}{oos_with_cost['max_dd']:>14.1%}")
    print(f"{'Net Value':<25}{best_result['net_value']:>14.2f}x{oos_with_cost['net_value']:>13.2f}x")
    print(f"{'Trading Cost':<25}{best_result['total_cost']:>14.2%}{oos_with_cost['total_cost']:>14.2%}")
    print(f"{'Avg Turnover':<25}{best_result['avg_turnover']:>14.1%}{oos_with_cost['avg_turnover']:>14.1%}")
    
    # 稳定性评估
    print(f"\n  Stability Assessment:")
    sharpe_decay = (best_result['sharpe'] - oos_with_cost['sharpe']) / best_result['sharpe'] if best_result['sharpe'] > 0 else 0
    if sharpe_decay < 0.2:
        stability = "[+] EXCELLENT (Sharpe decay < 20%)"
    elif sharpe_decay < 0.4:
        stability = "[o] GOOD (Sharpe decay 20-40%)"
    elif sharpe_decay < 0.6:
        stability = "[~] MODERATE (Sharpe decay 40-60%)"
    else:
        stability = "[-] POOR (Sharpe decay > 60%)"
    print(f"    Sharpe Decay: {sharpe_decay:.1%}")
    print(f"    Assessment: {stability}")
    
    # 输出相关性矩阵
    print(f"\n  Correlation matrix of selected factors:")
    selected_corr = corr_matrix.loc[best_combo, best_combo]
    print(selected_corr.round(2).to_string())
    
    off_diag = selected_corr.values[np.triu_indices(len(selected_corr), k=1)]
    print(f"\n  Max correlation in selection: {np.max(np.abs(off_diag)):.2f}")
    
    # 输出FACTOR_CONFIG格式
    print("\n" + "=" * 70)
    print("FACTOR_CONFIG (copy to crypto_strategy.py):")
    print("=" * 70)
    print("FACTOR_CONFIG = {")
    for f in sorted_best:
        row = qualified_df[qualified_df['factor'] == f].iloc[0]
        freq = 'daily' if row['type'] == 'Daily' else 'intraday'
        print(f"    '{f}': FactorInfo('{f}', 'Auto', '{f}', 1, '{freq}'),")
    print("}")
    
    return best_combo, best_result, oos_results


if __name__ == "__main__":
    best_factors, in_sample_result, oos_results = main()