"""
加密货币多因子截面策略 (流动性过滤版本)

策略逻辑：
1. 从预计算的因子文件加载因子值
2. 每天动态筛选quote_volume前40%的标的（Q1+Q2流动性档位）
3. 在筛选后的标的中，等权组合生成信号
4. 做多top 20%,做空bottom 20%

因子构成 (8个因子):
=== Daily因子 ===
- lower_shadow            - 下影线

=== Intraday因子 ===
- trade_size_skew         - 单笔成交量偏度
- max_buy_pressure_1h     - 最大买压(1小时)
- market_r2               - 市场R²
- early_main_diff         - 早盘vs后续收益差
- info_entropy            - 信息分布熵
- amihud_illiq_log        - Amihud非流动性(log)
- max_drawdown_intra      - 日内最大回撤
"""

import glob
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class BacktestResult:
    """回测结果数据类"""
    total_net_value: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    cumulative_returns: pd.Series
    daily_returns: pd.Series
    # 交易成本相关
    total_cost: float = 0.0              # 累计交易成本
    avg_daily_turnover: float = 0.0      # 日均换手率
    cost_return_ratio: float = 0.0       # 成本/收益比
    sharpe_before_cost: float = 0.0      # 扣费前Sharpe
    daily_costs: pd.Series = None        # 每日交易成本序列
    # 参与率限制相关
    avg_utilization: float = 1.0         # 平均资金利用率（因参与率限制）


@dataclass
class FactorInfo:
    """因子信息数据类"""
    name: str
    category: str
    description: str
    direction: int
    freq: str


# ============================================================
# 因子配置 (8个因子)
# ============================================================

FACTOR_CONFIG = {
    'trade_size_skew': FactorInfo('trade_size_skew', 'Microstructure', '单笔成交量偏度', 1, 'intraday'),
    'max_buy_pressure_1h': FactorInfo('max_buy_pressure_1h', 'Microstructure', '最大买压(1小时)', 1, 'intraday'),
    'market_r2': FactorInfo('market_r2', 'Correlation', '市场R²', 1, 'intraday'),
    'early_main_diff': FactorInfo('early_main_diff', 'Session', '早盘vs后续收益差', 1, 'intraday'),
    'info_entropy': FactorInfo('info_entropy', 'Information', '信息分布熵', 1, 'intraday'),
    'amihud_illiq_log': FactorInfo('amihud_illiq_log', 'Liquidity', 'Amihud非流动性(log)', 1, 'intraday'),
    'lower_shadow': FactorInfo('lower_shadow', 'Technical', '下影线', 1, 'daily'),
    'max_drawdown_intra': FactorInfo('max_drawdown_intra', 'Volatility', '日内最大回撤', 1, 'intraday'),
}


# ============================================================
# 工具函数
# ============================================================

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score 标准化"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)


# ============================================================
# 因子加载器（从CSV文件加载预计算因子）
# ============================================================

class FactorLoader:
    """因子加载器 - 统一从CSV文件加载预计算的因子值"""

    def __init__(self, daily_factors_file: str, intraday_factors_file: str):
        self.daily_factors_file = daily_factors_file
        self.intraday_factors_file = intraday_factors_file

    def load_all_factors(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载所有因子（只加载FACTOR_CONFIG中定义的因子）
        
        Args:
            normalize: 是否进行截面标准化
        
        Returns:
            Dict[factor_name, DataFrame(date x symbol)]
        """
        factors = {}
        
        # 加载日度因子
        if Path(self.daily_factors_file).exists():
            factors.update(self._load_from_csv(self.daily_factors_file))
            print(f"Loaded daily factors from: {self.daily_factors_file}")
        else:
            print(f"Warning: Daily factors file not found: {self.daily_factors_file}")
        
        # 加载日内因子
        if Path(self.intraday_factors_file).exists():
            factors.update(self._load_from_csv(self.intraday_factors_file))
            print(f"Loaded intraday factors from: {self.intraday_factors_file}")
        else:
            print(f"Warning: Intraday factors file not found: {self.intraday_factors_file}")
        
        # 截面标准化
        if normalize:
            factors = {k: zscore(v) for k, v in factors.items()}
        
        print(f"Total factors loaded: {list(factors.keys())}")
        return factors
    
    def _load_from_csv(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """从单个CSV文件加载因子"""
        factors = {}
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            
            # 只加载FACTOR_CONFIG中定义的因子
            factor_cols = [c for c in df.columns 
                          if c not in ['symbol', 'date'] and c in FACTOR_CONFIG]
            
            for col in factor_cols:
                try:
                    wide = df.pivot(index='date', columns='symbol', values=col)
                    factors[col] = wide
                except Exception as e:
                    print(f"  Warning: Failed to pivot {col} - {e}")
            
            print(f"  Loaded factors: {factor_cols}")
            
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
        
        return factors


# ============================================================
# 因子分析器
# ============================================================

class FactorAnalyzer:
    def __init__(self, factors: Dict[str, pd.DataFrame], forward_returns: pd.DataFrame):
        self.factors = factors
        self.forward_returns = forward_returns
    
    def calc_ic(self, name: str) -> pd.Series:
        fc = self.factors.get(name)
        if fc is None:
            return pd.Series()
        idx = fc.index.intersection(self.forward_returns.index)
        cols = fc.columns.intersection(self.forward_returns.columns)
        ic_list = []
        for d in idx:
            f_val = fc.loc[d, cols].values
            r_val = self.forward_returns.loc[d, cols].values
            valid = ~(np.isnan(f_val) | np.isnan(r_val))
            if valid.sum() >= 20:
                ic, _ = stats.spearmanr(f_val[valid], r_val[valid])
                ic_list.append((d, ic))
        return pd.Series(dict(ic_list))
    
    def analyze_all_factors(self) -> pd.DataFrame:
        res = []
        for name in self.factors:
            ic = self.calc_ic(name)
            if len(ic) == 0:
                continue
            res.append({
                'factor': name, 'ic_mean': ic.mean(), 'ic_std': ic.std(),
                'ic_ir': ic.mean() / (ic.std() + 1e-8), 'ic_positive_ratio': (ic > 0).mean(),
            })
        return pd.DataFrame(res).set_index('factor')
    
    def calc_correlation_matrix(self) -> pd.DataFrame:
        ic_dict = {k: self.calc_ic(k) for k in self.factors}
        ic_dict = {k: v for k, v in ic_dict.items() if len(v) > 0}
        return pd.DataFrame(ic_dict).corr()

# ============================================================
# 回测引擎
# ============================================================

class Backtester:
    def __init__(self, forward_returns: pd.DataFrame, quote_volume: pd.DataFrame = None,
                 close_price: pd.DataFrame = None, volatility_window: int = 20):
        """
        Args:
            forward_returns: 前向收益率（T日因子对应T+1日持仓收益，open-to-open）
            quote_volume: 成交额数据
            close_price: 收盘价数据（用于计算历史波动率）
            volatility_window: 波动率计算窗口（默认20天）
        """
        self.forward_returns = forward_returns
        self.quote_volume = quote_volume
        self.volatility_window = volatility_window
        
        # 滚动波动率（用于市场冲击模型）
        if close_price is not None:
            close_ret = close_price.pct_change()
            self.rolling_volatility = close_ret.rolling(
                window=volatility_window, min_periods=10
            ).std()
        else:
            # 兼容旧调用方式
            self.rolling_volatility = forward_returns.shift(2).rolling(
                window=volatility_window, min_periods=10
            ).std()
    
    @staticmethod
    def calc_trading_cost(participation_rate: float, 
                          daily_volatility: float = 0.05,
                          taker_fee: float = 0.0005,
                          impact_coef: float = 0.12) -> float:
        """
        计算单次交易的总成本（单边）
        
        Args:
            participation_rate: 参与率 = 交易金额 / 市场成交额
            daily_volatility: 日波动率（默认5%）
            taker_fee: Taker手续费率（默认0.05%）
            impact_coef: 市场冲击系数
        
        Returns:
            total_cost_rate: 总成本比例（单边）
            
        市场冲击模型 (Square Root Law):
            Impact = c * σ * √(participation_rate)
        """
        if participation_rate <= 0:
            return taker_fee
        
        impact = impact_coef * daily_volatility * np.sqrt(participation_rate)
        impact = np.clip(impact, 0, 0.05)
        
        return taker_fee + impact
    
    def run(self, signals: pd.DataFrame, long_pct: float = 0.2, short_pct: float = 0.2,
            weight_method: str = 'sqrt_volume', leverage: float = 1.0,
            enable_cost: bool = True, taker_fee: float = 0.0005,
            impact_coef: float = 0.12, aum: float = 10_000_000,
            max_participation: float = 0.05,
            liquidity_filter: float = 0.4,
            funding_adj: dict = None) -> BacktestResult:
        """
        运行回测
        
        Args:
            signals: 因子信号
            long_pct: 多头比例
            short_pct: 空头比例
            weight_method: 权重方法 'equal' | 'sqrt_volume'
            leverage: 杠杆倍数，1.0表示多空各50%，2.0表示多空各100%
            enable_cost: 是否计算交易成本
            taker_fee: Taker手续费率
            impact_coef: 市场冲击系数
            aum: 策略资金规模（USDT），用于计算市场冲击
            max_participation: 最大参与率限制（默认5%）
            liquidity_filter: 流动性过滤比例，只保留quote_volume前N%的标的
                             默认0.4=前40%（即Q1+Q2），设为1.0则不过滤
            funding_adj: dict[(symbol, date)] -> 当日结算费率之和，用于收益扣除（Layer 2）
        """
        idx = signals.index.intersection(self.forward_returns.index)
        cols = signals.columns.intersection(self.forward_returns.columns)
        sig = signals.loc[idx, cols]
        fwd = self.forward_returns.loc[idx, cols]
        
        # 如果使用volume加权，对齐quote_volume
        if self.quote_volume is not None:
            qv = self.quote_volume.reindex(index=idx, columns=cols)
        else:
            qv = None
        
        daily_rets_before_cost = []
        daily_rets_after_cost = []
        daily_costs = []
        daily_turnovers = []
        daily_utilizations = []
        dates = []
        
        # 上一日持仓权重 (symbol -> weight)
        prev_long_weights = {}
        prev_short_weights = {}

        def _side_cost(weights_new, weights_old, v_ser, vol_today):
            """计算单侧换手成本和换手率"""
            cost = turnover = 0.0
            for sym in set(weights_new) | set(weights_old):
                delta_w = abs(weights_new.get(sym, 0) - weights_old.get(sym, 0))
                if delta_w > 1e-8 and sym in v_ser.index:
                    trade_value = delta_w * aum * 0.5 * leverage
                    participation = trade_value / (v_ser[sym] + 1e-8)
                    sym_vol = 0.05
                    if vol_today is not None and sym in vol_today.index:
                        vv = vol_today[sym]
                        if not pd.isna(vv) and vv > 0:
                            sym_vol = vv
                    cost_rate = self.calc_trading_cost(participation, sym_vol, taker_fee, impact_coef)
                    cost += delta_w * cost_rate * 0.5 * leverage
                    turnover += delta_w
            return cost, turnover

        
        for d in idx:
            s, r = sig.loc[d], fwd.loc[d]
            valid = ~(s.isna() | r.isna())
            
            # 如果有volume数据，也要求volume有效
            if qv is not None:
                v = qv.loc[d]
                valid = valid & ~v.isna() & (v > 0)
            else:
                v = None
            
            if valid.sum() < 20:
                continue
            
            # ========== 流动性过滤：只保留quote_volume前N%的标的 ==========
            if liquidity_filter < 1.0 and qv is not None and v is not None:
                # 计算当天有效标的的quote_volume排名
                valid_symbols = valid[valid].index
                vol_valid = v[valid_symbols]
                
                # 按quote_volume降序排名，取前liquidity_filter比例
                vol_rank = vol_valid.rank(ascending=False, pct=True)
                liquidity_mask = vol_rank <= liquidity_filter
                
                # 更新valid，只保留高流动性标的
                filtered_symbols = vol_rank[liquidity_mask].index
                valid = pd.Series(False, index=valid.index)
                valid[filtered_symbols] = True
                
                # 重新检查数量
                if valid.sum() < 10:
                    continue
            
            # 计算当日目标权重
            n = valid.sum()
            ln, sn = int(n * long_pct), int(n * short_pct)
            ranks = s[valid].rank(ascending=False)
            
            long_mask = ranks <= ln
            short_mask = ranks > (n - sn)
            
            capital_per_side = aum * leverage / 2
            if weight_method == 'sqrt_volume' and qv is not None:
                weights = np.sqrt(v[valid])
                
                # 多头权重
                long_w = weights[long_mask].copy()
                long_w = long_w / long_w.sum() if long_w.sum() > 0 else long_w
                
                # 空头权重
                short_w = weights[short_mask].copy()
                short_w = short_w / short_w.sum() if short_w.sum() > 0 else short_w
                
            else:
                # 等权
                long_symbols = s[valid][long_mask].index.tolist()
                short_symbols = s[valid][short_mask].index.tolist()
                
                long_w = pd.Series({sym: 1.0/len(long_symbols) for sym in long_symbols}) if long_symbols else pd.Series()
                short_w = pd.Series({sym: 1.0/len(short_symbols) for sym in short_symbols}) if short_symbols else pd.Series()
            
            # 应用参与率限制（cap权重）
            if max_participation > 0 and qv is not None and len(long_w) > 0:
                def _cap(w):
                    sym_vols = v.reindex(w.index).fillna(0)
                    max_ws = (sym_vols * max_participation / capital_per_side).where(sym_vols > 0, other=np.inf)
                    return w.clip(upper=max_ws)
                long_w = _cap(long_w)
                short_w = _cap(short_w)
                long_utilization = long_w.sum()
                short_utilization = short_w.sum()
            else:
                long_utilization = 1.0
                short_utilization = 1.0
            
            daily_utilizations.append((long_utilization + short_utilization) / 2)
            long_weights = dict(zip(long_w.index, long_w.values)) if len(long_w) > 0 else {}
            short_weights = dict(zip(short_w.index, short_w.values)) if len(short_w) > 0 else {}
            
            # 计算收益，含资金费率成本
            def _side_ret(weights_dict):
                if not weights_dict:
                    return 0.0
                ws = pd.Series(weights_dict)
                rets = r.reindex(ws.index).fillna(0)
                if funding_adj:
                    rets = rets - pd.Series({sym: funding_adj.get((sym, d), 0) for sym in ws.index})
                return float((rets * ws).sum())

            long_r = _side_ret(long_weights)
            short_r = _side_ret(short_weights)
            
            # 应用杠杆
            daily_ret_before = (long_r - short_r) * leverage / 2
            if np.isnan(daily_ret_before):
                daily_ret_before = 0.0
            daily_rets_before_cost.append(daily_ret_before)
            
            if enable_cost and qv is not None and v is not None:
                vol_today = self.rolling_volatility.loc[d] if d in self.rolling_volatility.index else None
                long_cost, long_turnover = _side_cost(long_weights, prev_long_weights, v, vol_today)
                short_cost, short_turnover = _side_cost(short_weights, prev_short_weights, v, vol_today)
                total_cost = long_cost + short_cost
                avg_turnover = (long_turnover / 2 + short_turnover / 2) / 2
                
                daily_costs.append(total_cost)
                daily_turnovers.append(avg_turnover)
                daily_ret_after = daily_ret_before - total_cost
                if np.isnan(daily_ret_after):
                    daily_ret_after = daily_ret_before
            else:
                daily_costs.append(0.0)
                daily_turnovers.append(0.0)
                daily_ret_after = daily_ret_before
            
            if np.isnan(daily_ret_after):
                daily_ret_after = 0.0
            
            daily_rets_after_cost.append(daily_ret_after)
            dates.append(d)
            
            prev_long_weights = long_weights
            prev_short_weights = short_weights
        
        daily_rets_before = pd.Series(daily_rets_before_cost, index=dates)
        daily_rets = pd.Series(daily_rets_after_cost, index=dates)
        daily_costs_series = pd.Series(daily_costs, index=dates)
        
        cum = (1 + daily_rets).cumprod()
        cum_before = (1 + daily_rets_before).cumprod()
        
        nv = cum.iloc[-1] if len(cum) > 0 else 1.0
        nv_before = cum_before.iloc[-1] if len(cum_before) > 0 else 1.0
        
        n_years = len(daily_rets) / 365
        ann_ret = (nv ** (1 / n_years)) - 1 if n_years > 0 else 0
        ann_vol = daily_rets.std() * np.sqrt(365)
        sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(365)
        
        ann_ret_before = (nv_before ** (1 / n_years)) - 1 if n_years > 0 else 0
        sharpe_before = daily_rets_before.mean() / (daily_rets_before.std() + 1e-8) * np.sqrt(365)
        
        cum_clean = cum.dropna()
        if len(cum_clean) > 0:
            rm = np.maximum.accumulate(cum_clean.values)
            mdd = np.max((rm - cum_clean.values) / (rm + 1e-8))
        else:
            mdd = 0.0
        
        wins = daily_rets[daily_rets > 0]
        losses = daily_rets[daily_rets < 0]
        wr = len(wins) / len(daily_rets) if len(daily_rets) > 0 else 0
        plr = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
        
        total_cost = daily_costs_series.sum()
        avg_turnover = np.mean(daily_turnovers) if daily_turnovers else 0
        total_return = nv - 1
        cost_return_ratio = total_cost / total_return if total_return > 0 else 0
        
        avg_util = np.mean(daily_utilizations) if daily_utilizations else 1.0
        
        return BacktestResult(
            total_net_value=nv,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=wr,
            profit_loss_ratio=plr,
            cumulative_returns=cum,
            daily_returns=daily_rets,
            total_cost=total_cost,
            avg_daily_turnover=avg_turnover,
            cost_return_ratio=cost_return_ratio,
            sharpe_before_cost=sharpe_before,
            daily_costs=daily_costs_series,
            avg_utilization=avg_util,
        )


# ============================================================
# 数据加载辅助函数
# ============================================================


def daily_winsorize(df: pd.DataFrame, lower: int = 1, upper: int = 99) -> pd.DataFrame:
    """日度 Winsorize 处理（截面向量化）"""
    lo = df.quantile(lower / 100, axis=1)
    hi = df.quantile(upper / 100, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def load_returns_from_price(price_data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    从原始价格数据目录加载收益率、quote_volume、close价格和市场指数
    （CSV格式，每个标的一个文件）
    
    时序逻辑（open-to-open）：
        T日 24:00   → 计算T日因子（需要T日全天数据）
        T+1日 00:00 → 根据T日因子，以 open[T+1] 开仓
        T+2日 00:00 → 根据T+1日因子调仓，以 open[T+2] 平仓
        
        因此: fwd_ret[T] = open[T+2] / open[T+1] - 1
              即T日因子对应的持仓收益
    
    Returns:
        (forward_returns, quote_volume, close_price, market_index)
    """
    files = glob.glob(f"{price_data_dir}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files in {price_data_dir}")
    
    dfs = []
    for f in files:
        symbol = Path(f).stem
        try:
            df = pd.read_csv(f)
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            dfs.append(df)
        except Exception as e:
            print(f"Warning: {f} - {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['open_time'] = pd.to_datetime(combined['open_time']).dt.normalize()
    combined['open'] = pd.to_numeric(combined['open'], errors='coerce')
    combined['close'] = pd.to_numeric(combined['close'], errors='coerce')
    combined['quote_volume'] = pd.to_numeric(combined['quote_volume'], errors='coerce')
    
    open_ = combined.pivot_table(index='open_time', columns='symbol', values='open', aggfunc='first')
    close = combined.pivot_table(index='open_time', columns='symbol', values='close', aggfunc='last')
    quote_volume = combined.pivot_table(index='open_time', columns='symbol', values='quote_volume', aggfunc='sum')
    
    # 计算open-to-open前向收益率（用于回测实际持仓收益）
    # fwd_ret[T] = open[T+2] / open[T+1] - 1
    # 即：T日因子 → T+1日open开仓 → T+2日open平仓
    open_ret = open_.pct_change()
    fwd_ret = daily_winsorize(open_ret.shift(-2), 1, 99)
    
    # 计算流动性加权市场指数（使用close价格）
    weighted_price = (close * quote_volume).sum(axis=1) / quote_volume.sum(axis=1)
    market_index = weighted_price / weighted_price.iloc[0]  # 归一化到1
    market_index.name = 'Market Index'
    
    print(f"Loaded returns (open-to-open): {len(fwd_ret)} days, {len(fwd_ret.columns)} symbols")
    
    return fwd_ret, quote_volume, close, market_index


# ============================================================
# 资金费率 (Funding Rate)
# ============================================================

def _load_funding_settlements(funding_dir="./funding_rates"):
    """加载所有结算数据，按 (symbol, holding_date) 分组。

    Returns:
        dict[(symbol, holding_date)] -> list of (calc_time, funding_rate)
    """
    funding_path = Path(funding_dir)
    all_records = []
    for f in funding_path.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            if 'funding_rate' not in df.columns:
                continue
            df = df[['calc_time', 'funding_rate']].copy()
            df['symbol'] = f.stem
            df['calc_time'] = pd.to_datetime(df['calc_time'])
            all_records.append(df)
        except Exception:
            pass
    if not all_records:
        return {}
    big = pd.concat(all_records, ignore_index=True)
    big['holding_date'] = big['calc_time'].dt.normalize()
    midnight = big['calc_time'].dt.hour == 0
    big.loc[midnight, 'holding_date'] -= pd.Timedelta(days=1)
    big = big.sort_values(['symbol', 'holding_date', 'calc_time'])

    groups = {}
    for (sym, hdate), grp in big.groupby(['symbol', 'holding_date']):
        groups[(sym, hdate)] = list(zip(grp['calc_time'], grp['funding_rate']))
    return groups




def precompute_funding_costs(funding_dir="./funding_rates", verbose=False):
    """预计算每个 (symbol, factor_date) 的资金费率总成本。

    对所有持仓的 funding 调整: ret_val -= funding_adj
    (多头付正费率、空头收正费率，这个公式对两个方向都正确)

    Returns:
        dict[(symbol, factor_date)] → 当日所有结算费率之和
    """
    settle_groups = _load_funding_settlements(funding_dir)
    if verbose:
        print(f"  {len(settle_groups):,} settlement groups loaded")
    funding_adj = {}
    for (sym, hdate), settlements in settle_groups.items():
        factor_date = hdate - pd.Timedelta(days=1)
        funding_adj[(sym, factor_date)] = sum(r for _, r in settlements)
    return funding_adj


def precompute_expected_funding_rate(funding_dir="./funding_rates", window=14,
                                      method='ewm', verbose=False):
    """预计算每个 (symbol, factor_date) 的预期资金费率。

    用于成本调整信号: adjusted_signal = signal - penalty * expected_rate
    - expected_rate > 0 → 多头成本高 → 信号降低 → 排名下降
    - expected_rate < 0 → 空头成本高 → 信号升高 → 排名上升

    Args:
        funding_dir: 资金费率数据目录（或多个目录的 list）
        window: 滚动窗口天数（SMA 为窗口大小，EWM 为 span）
        method: 'sma'（简单移动平均）或 'ewm'（指数加权移动平均）
        verbose: 是否打印统计信息

    Returns:
        pd.DataFrame: 宽表 (factor_date × symbol)，值为带符号的预期日均费率
    """
    dirs = [funding_dir] if isinstance(funding_dir, (str, Path)) else funding_dir
    settle_groups = {}
    for d in dirs:
        settle_groups.update(_load_funding_settlements(d))

    if not settle_groups:
        if verbose:
            print("  [ExpectedFR] No funding data found")
        return pd.DataFrame()

    records = []
    for (sym, hdate), settlements in settle_groups.items():
        factor_date = hdate - pd.Timedelta(days=1)
        records.append({'symbol': sym, 'factor_date': factor_date,
                        'daily_rate_sum': sum(r for _, r in settlements)})

    df = pd.DataFrame(records).sort_values(['symbol', 'factor_date'])
    # shift(1): daily_rate_sum[factor_date T] 对应的是 holding_date T+1 的结算（未来数据），
    # 生成 factor_date T 的信号时尚不可知。shift 后只使用已实现的结算数据。
    if method == 'ewm':
        df['expected_rate'] = (
            df.groupby('symbol')['daily_rate_sum']
            .transform(lambda x: x.shift(1).ewm(span=window, min_periods=2).mean())
        )
    else:
        df['expected_rate'] = (
            df.groupby('symbol')['daily_rate_sum']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )

    wide = df.dropna(subset=['expected_rate']).pivot_table(
        index='factor_date', columns='symbol', values='expected_rate', aggfunc='first'
    )

    if verbose:
        print(f"  [ExpectedFR] method={method}, window={window}d, shape={wide.shape}")
        print(f"  [ExpectedFR] mean={wide.mean().mean():.6f}, "
              f"std={wide.stack().std():.6f}")

    return wide

