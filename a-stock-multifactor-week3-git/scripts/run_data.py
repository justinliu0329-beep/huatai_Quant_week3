import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from src.data_fetch import fetch_stock_list, batch_download, fetch_benchmark
from src.utils import log


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    log("① 拉取全 A 股列表 ...")
    pool = fetch_stock_list()
    codes = pool["code"].tolist()
    if limit:
        codes = codes[:limit]
    log(f"   待下载 {len(codes)} 只")

    log("② 下载日线（前复权）...")
    batch_download(codes, kind="daily")

    log("③ 下载财务指标 ...")
    batch_download(codes, kind="fin")

    log("④ 下载基准指数 ...")
    fetch_benchmark()

    log("✅ 数据下载完成")


if __name__ == "__main__":
    main()
