import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
C.N_WORKERS = 8
from src.data_fetch import fetch_stock_list, batch_download
from src.utils import log


def main():
    pool = fetch_stock_list()
    codes = pool["code"].tolist()
    log(f"财务指标下载：{len(codes)} 只（同花顺源，8 线程）")
    batch_download(codes, kind="fin")
    log("✅ 财务下载完成")


if __name__ == "__main__":
    main()
