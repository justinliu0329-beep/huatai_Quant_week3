# A 股多因子短线交易策略研究

> 三周实习项目交付 ｜ 作者：柳佳成 ｜ 数据源：AKShare（新浪行情 / 同花顺财务）

---

## 一、目录结构

```
a-stock-multifactor/
├── config.py               # 全局配置：样本区间/股票池规则/因子参数
├── requirements.txt
├── data/                   # 数据与缓存（运行后生成）
│   ├── cache/daily/         # 每只票日线缓存 (parquet)
│   ├── cache/fin/           # 每只票财务缓存 (parquet)
│   ├── stock_pool_raw.csv   # 全A股列表 + 上市日期 + 板块 + 行业
│   ├── benchmark.csv        # 沪深300 基准
│   ├── price_panel_clean.parquet
│   ├── financial_panel.parquet
│   └── factor_panel*.parquet
├── src/
│   ├── utils.py             # 重试 / 缓存 / 日期 / 日志
│   ├── data_fetch.py        # 数据获取（并发 + 断点续传）
│   ├── data_clean.py        # 清洗 + 股票池剔除 + 防未来函数对齐 + 切分
│   ├── factors.py           # 因子构建（技术类 + 基本面类）
│   ├── factor_test.py       # [W1] IC/IR + 分层回测 + 相关性/正交化 + 绘图
│   ├── composite.py         # [W2] IC 加权合成 + 选股 + 仓位约束 + 目标权重生成
│   ├── backtest.py          # [W2] VectorBT 回测引擎 + 自研成本核算
│   ├── performance.py       # [W2] 绩效指标 + 净值/回撤/月度图
│   ├── volatility.py        # [W3] GARCH 波动率预测 + 动态敞口 + 趋势过滤
│   └── stress.py            # [W3] 尾部极端行情压力测试
├── scripts/
│   ├── run_data.py          # 下载日线行情
│   ├── run_financial.py     # 下载财务指标
│   ├── run_factors.py       # 构建清洗面板 + 因子面板
│   ├── run_test.py          # [W1] ★ 单因子有效性检验
│   ├── run_backtest.py      # [W2] ★ 静态多因子基准策略回测
│   ├── run_dynamic.py       # [W3] ★ 动态风控策略 + 对比 + 压力测试
│   └── qmt_signals.py       # [W2] 加分：QMT 周度调仓信号
├── docs/
│   └── week2_backtest.md    # [W2] 回测参数 + 风控规则实现说明
├── output/
│   ├── figures/             # 图表 png（IC/分层/相关/净值/回撤/月度/对比/敞口）
│   └── tables/              # 统计表 csv
└── reports/
    ├── week1_factor_report.md      # ★ Week1 因子检验报告
    ├── week2_static_backtest.md    # ★ Week2 静态策略回测报告
    └── week3_research_report.md    # ★ Week3 完整量化研究报告
```

## 二、快速开始

```bash
python3 -m venv ~/akshare-env
source ~/akshare-env/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

python scripts/run_data.py        # ① 下载日线（全市场，新浪源，约 20 分钟）
python scripts/run_financial.py   # ② 下载财务（同花顺源，约 40 分钟）
python scripts/run_factors.py     # ③ 构建清洗面板 + 因子面板
python scripts/run_test.py        # ④ 单因子检验 → 出图 + 报告
python scripts/run_backtest.py    # ⑤ [W2] 静态多因子基准策略回测
python scripts/run_dynamic.py     # ⑥ [W3] 动态风控策略 + 对比 + 压力测试
```

## 三、硬性约束实现

| 约束 | 实现位置 |
|---|---|
| 训练 2020-24 / 验证 2025 / 测试 2026.1-8 | `config.SPLITS`、`data_clean.split_panel` |
| 剔除 ST/*ST/退市 | `data_clean.apply_pool_rules`（按证券简称） |
| 剔除上市不足 120 自然日新股 | `apply_pool_rules`（date − list_date < 120） |
| 剔除日均成交额 < 5000 万 | `apply_pool_rules`（全样本日均成交额） |
| 周度调仓 | `factor_test.weekly_rebalance_dates`（每周最后交易日） |

## 四、测试

```bash
python -m pytest          # 运行 tests/ 套件（18 用例，覆盖因子/约束/回测/清洗）
```

覆盖：因子 IC 加权合成、仓位约束（持仓/单票/行业上限）、目标权重生成（含止盈止损）、
回测引擎不变量（净值/成本）、因子计算的**防未来函数**、财务披露日对齐、股票池剔除规则。

## 五、关键工程与风控设计（防未来函数）

1. **数据源选型**：eastmoney 行情接口实测被 IP 限流（连续请求后成功率掉到 0），改用
   **新浪源 `stock_zh_a_daily`（前复权）**，10 线程并发全市场约 20 分钟；
   财务用**同花顺摘要**（新浪财务接口实测返回空）。
2. **财务披露日对齐**：财务因子不按"报告期"直接用，而映射到**法定披露截止日**
   （Q1/年报→4-30，半年报→8-31，Q3→10-31）之后才生效（`data_clean.financial_available_date`）。
3. **价量因子**：全部用 `shift`/`rolling`，t 日因子仅含 `<=t` 数据。
4. **停牌处理**：`volume<=0` 或收盘缺失的交易日剔除。
5. **去极值标准化**：横截面 MAD 去极值（5 倍 MAD）+ z-score。
6. **缓存与断点续传**：原始数据落地 parquet，重复运行自动跳过已下载。

## 六、因子清单（4 技术 + 4 基本面）

| 类别 | 因子 | 含义 | 方向直觉 |
|---|---|---|---|
| 技术 | momentum_20 | 过去 20 日收益率 | 动量/反转 |
| 技术 | reversal_5 | 过去 5 日收益率 | 短线反转 |
| 技术 | volatility_20 | 过去 20 日日收益标准差 | 低波溢价 |
| 技术 | turnover_20 | 过去 20 日平均换手率 | 低换手溢价 |
| 基本面 | roe | 净资产收益率 | 质量 |
| 基本面 | gross_margin | 销售毛利率 | 质量 |
| 基本面 | pb | 市净率 | 价值 |
| 基本面 | growth | 净利润同比增长率 | 成长 |

## 七、Week1 主要结论（详见 reports/week1_factor_report.md）

- 训练集（2020-2024）周度 Rank IC 检验，**5 个因子显著**（|t|>2）：
  波动率_20(t=-6.85)、换手_20(t=-5.77)、PB(t=-5.18)、动量_20(t=-4.78)、反转_5(t=-3.53)。
- 全样本 5 组分层回测显示各因子具收益排序能力（低估值、低波动、低换手组显著占优）。
- 因子相关性：波动率×换手相关 0.72（>0.7），触发 Gram-Schmidt 正交化剔除冗余。
- 净利润增长率/ROE/毛利率 训练集 IC 不显著，暂不纳入多因子模型（详见报告第六节）。

## 八、Week2：静态多因子基准策略

基于 Week1 的 5 个有效因子，**IC 加权**合成综合分，周度选股，用 **VectorBT** 搭建回测
（自研因子打分/调仓/仓位约束/成本/止盈止损）。

```bash
python scripts/run_backtest.py     # 静态策略回测 → 净值/回撤/月度图 + 指标表
python scripts/qmt_signals.py      # 加分：输出 QMT 周度调仓信号
```

**实测绩效**：

| 指标 | 全样本 2020-26 | 验证集 2025 | 测试集 2026 |
|---|---|---|---|
| 年化收益率 | 1.79% | 25.5% | -4.96% |
| 年化波动率 | 17.8% | 18.4% | 22.9% |
| 夏普比率 | 0.10 | 1.39 | -0.22 |
| 最大回撤 | -36.5% | -13.8% | -18.9% |
| 超额(vs沪深300)累计 | -6.0% | +2.7% | -0.02% |

- 累计交易成本 **44.7 万**（占初始资金 45%）——周度调仓换手高，成本拖累显著。
- 该策略**作为对照组**，供 Week3 的波动率动态风控策略对比。
- 详见 `reports/week2_static_backtest.md` 与 `docs/week2_backtest.md`（回测参数、风控实现说明）。

## 九、Week3：波动率动态风控策略

**双层架构**：第一层多因子选股（Week2），第二层风险约束 —— GARCH 波动率动态仓位
（防波动尖峰）+ 趋势过滤（防慢熊），两者取较小值。

```bash
python scripts/run_dynamic.py   # 动态策略 + 静态对比 + 尾部压力测试 → 研究报告
```

**优化效果（全样本 2020-2026）**：

| 指标 | 静态（对照） | **动态风控** |
|---|---|---|
| 年化收益率 | 1.79% | **2.48%** |
| 年化波动率 | 17.76% | **16.22%** |
| 夏普比率 | 0.1007 | **0.1527** |
| 最大回撤 | -36.52% | **-28.34%** |
| Calmar率 | 0.0490 | **0.0874** |
| 超额收益(vs沪深300) | -6.03% | **-1.78%** |

- **核心发现**：① 纯 GARCH 波动率降仓对**慢熊**无效（慢熊中波动率不极端）→ 迭代叠加
  趋势过滤后回撤显著改善；② 组合策略在**所有指标上全面优于静态**，且验证集（平稳行情）
  超额衰减 19%（< 20% 阈值，达标）。
- 详见 `reports/week3_research_report.md`（含压力测试、模型局限性、实盘风险、改进方向）。
