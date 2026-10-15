# -*- coding: utf-8 -*-
"""
项目全局配置：路径、样本区间、股票池规则、交易约束、因子检验参数。

作者：柳佳成（实习）
"""
import os

# ---------------- 路径 ----------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CACHE_DIR  = os.path.join(DATA_DIR, "cache")
DAILY_CACHE = os.path.join(CACHE_DIR, "daily")   # 每只票的日线缓存
FIN_CACHE   = os.path.join(CACHE_DIR, "fin")     # 每只票的财务缓存
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR    = os.path.join(OUTPUT_DIR, "tables")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _d in (DATA_DIR, CACHE_DIR, DAILY_CACHE, FIN_CACHE,
           OUTPUT_DIR, FIG_DIR, TAB_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- 样本区间划分（严格区分，杜绝数据窥探）----------------
# 训练：因子挖掘/参数拟合 | 验证：权重调试 | 测试：样本外，全程禁调参
TRAIN_START, TRAIN_END = "2020-01-01", "2024-12-31"
VALID_START, VALID_END = "2025-01-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-08-01"
# 数据下载总区间（含预热期，保证年初动量因子可算）
DATA_START,  DATA_END  = "2019-10-01", "2026-08-01"

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test":  (TEST_START,  TEST_END),
}

# ---------------- 股票池筛选规则 ----------------
BENCHMARK     = "sh000300"   # 业绩基准：沪深300
MIN_LISTED_DAYS = 120        # 上市不足 120 自然日的新股剔除
MIN_AVG_AMOUNT  = 5e7        # 日均成交额 < 5000 万（流动性不足）剔除

# ---------------- 交易与回测约束（贴近实盘）----------------
INIT_CAPITAL = 1_000_000     # 初始虚拟资金 100 万
COMMISSION   = 0.0003        # 佣金 0.03%（双边）
STAMP_TAX    = 0.0005        # 印花税 0.05%（仅卖出）
SLIPPAGE     = 0.0005        # 单边滑点 0.05%
MIN_HOLD, MAX_HOLD = 8, 20   # 每期持仓 8~20 只
MAX_WEIGHT_PER_STOCK    = 0.15   # 单票 ≤ 组合 15%
MAX_WEIGHT_PER_INDUSTRY = 0.30   # 单行业 ≤ 组合 30%

# ---------------- 因子检验参数 ----------------
N_QUANTILE     = 5           # 分层回测组数
IC_METHOD      = "spearman"  # Rank IC
FORWARD_PERIOD = 5           # 未来 5 个交易日收益（对应周度调仓）
CORR_THRESHOLD = 0.7         # 因子相关 |r|>0.7 判为高共线，触发正交化

# ---------------- Week3：波动率动态风控参数 ----------------
TARGET_VOL     = 0.20        # 目标年化波动率（市场常态水平，敞口基准）
EXPOSURE_FLOOR = 0.30        # 最低敞口（预测高波动/极端行情时降至 30%）
EXPOSURE_CAP   = 1.00        # 满仓上限（低波动平稳行情）
VOL_WINDOW     = 250         # 波动率估计回看窗口（交易日）
EXPOSURE_KNEE  = 1.8         # 分段敞口：预测波动率 < 1.8×目标 → 满仓（不牺牲正常行情）
EXPOSURE_FULL  = 3.5         # 分段敞口：预测波动率 > 3.5×目标 → 降至底线（仅极端波动触发）
TREND_MA       = 200         # 趋势过滤均线（市场在均线下方=趋势走坏，防慢熊）
TREND_FLOOR    = 0.80        # 趋势走坏时敞口底线

# ---------------- 网络/缓存 ----------------
MAX_RETRY  = 4
RETRY_SLEEP = 1.5
N_WORKERS  = 10              # 并发下载线程数（实测新浪源 10 线程全市场约 20 分钟）
