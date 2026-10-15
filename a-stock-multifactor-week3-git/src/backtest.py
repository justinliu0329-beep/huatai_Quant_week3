import numpy as np
import pandas as pd
import vectorbt as vbt

import config as C


def run_backtest(prices: pd.DataFrame, targets: pd.DataFrame,
                 init_cash: float = C.INIT_CAPITAL,
                 commission: float = C.COMMISSION, stamp: float = C.STAMP_TAX,
                 slippage: float = C.SLIPPAGE) -> dict:


    pf = vbt.Portfolio.from_orders(
        prices, size=targets, size_type="targetpercent",
        group_by=True, cash_sharing=True, init_cash=init_cash,
        fees=0.0, slippage=0.0, freq="D")
    gross_nav = pf.value()


    buy_rate = commission + slippage
    sell_rate = commission + stamp + slippage
    orders = pf.orders.records_readable.copy()
    if len(orders):
        side = orders["Side"].astype(str).str.lower()
        rate = np.where(side.str.contains("sell"), sell_rate, buy_rate)
        orders["成交额"] = orders["Size"].abs() * orders["Price"]
        orders["成本"] = orders["成交额"] * rate
        cost_by_day = orders.groupby("Timestamp")["成本"].sum()
        cum_cost = cost_by_day.cumsum().reindex(gross_nav.index).ffill().fillna(0.0)
    else:
        orders["成交额"] = []
        orders["成本"] = []
        cum_cost = pd.Series(0.0, index=gross_nav.index)

    net_nav = (gross_nav - cum_cost).clip(lower=0)
    net_nav.index = pd.DatetimeIndex(net_nav.index)

    return {"gross_nav": gross_nav, "net_nav": net_nav, "orders": orders,
            "pf": pf, "total_cost": float(cum_cost.iloc[-1]) if len(cum_cost) else 0.0}
