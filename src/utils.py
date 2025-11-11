# -*- coding: utf-8 -*-
import os
import random
import logging
import sys
import time
from pathlib import Path
import numpy as np

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def setup_dirs(project_root: Path, paths: dict):
    for key, rel in paths.items():
        p = project_root / rel
        p.mkdir(parents=True, exist_ok=True)

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # 文件
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(ch); logger.addHandler(fh)
    logger.info(f"日志文件：{log_path}")
    return logger

class RunnerTimer:
    def __init__(self): self._t = {}
    def tic(self, name): self._t[name] = time.time()
    def toc(self, name):
        if name in self._t:
            dt = time.time() - self._t[name]
            print(f"[计时] {name}: {dt:.2f}s")
            return dt

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

def summarize_outputs(project_root: Path, cfg: dict, cv_acc: float, elapsed: float, logger):
    p = cfg["paths"]
    logger.info("\n=== 运行摘要 ===")
    logger.info(f"耗时：{human_time(elapsed)}")
    logger.info(f"CV/Val 准确率：{cv_acc:.4f}")
    logger.info("输出与产物：")
    logger.info(f"- 模型与向量器：{(project_root / p['checkpoints_dir']).resolve()}")
    logger.info(f"- 评估图表/报告：{(project_root / p['output_dir']).resolve()}")
    logger.info(f"- 日志：{(project_root / p['logs_dir']).resolve()}")
