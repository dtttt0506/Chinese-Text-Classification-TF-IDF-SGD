# -*- coding: utf-8 -*-
import time
import yaml
from pathlib import Path

from src.utils import (
    set_global_seed, setup_dirs, setup_logger, RunnerTimer, summarize_outputs
)
from src.data import load_data_sets
from src.preprocess import build_vectorizer
from src.trainers import run_grid_search, run_sgd_training, run_fast_mode
from src.evaluate import evaluate_and_plot
from src.predict import save_label_space

def main():
    t0 = time.time()
    project_root = Path(__file__).resolve().parent

    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg.get("seed", 42))
    paths = cfg["paths"]
    setup_dirs(project_root, paths)
    logger = setup_logger(project_root / paths["logs_dir"])
    timer = RunnerTimer()

    try:
        logger.info("=== 启动任务 ===")
        logger.info(f"运行模式: {cfg['train']['mode']}")

        # 数据
        timer.tic("data_loading")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), label_names = load_data_sets(
            project_root, cfg
        )
        timer.toc("data_loading")

        # 训练
        mode = cfg["train"]["mode"]
        best_model = None
        cv_acc = 0.0

        if mode == "grid":
            best_model, best_name, cv_acc = run_grid_search(
                X_train, y_train, cfg, project_root,
                label_names=label_names, logger=logger, timer=timer
            )
        elif mode == "sgd":
            vectorizer = build_vectorizer(cfg)
            best_model, best_name, cv_acc = run_sgd_training(
                X_train, y_train, X_val, y_val, cfg, project_root,
                vectorizer=vectorizer, label_names=label_names, logger=logger, timer=timer
            )
        elif mode == "fast":
            best_model, best_name, cv_acc = run_fast_mode(
                X_train, y_train, X_val, y_val, cfg, project_root,
                label_names=label_names, logger=logger, timer=timer
            )
        else:
            raise ValueError("train.mode 仅支持 'fast' | 'sgd' | 'grid'")

        # 测试评估
        out_dir = project_root / paths["output_dir"]
        ckpt_dir = project_root / paths["checkpoints_dir"]
        evaluate_and_plot(
            best_model, X_test, y_test, label_names,
            out_dir=out_dir,
            save_confmat=cfg["viz"].get("save_confusion_matrix", True),
            logger=logger
        )
        save_label_space(best_model, ckpt_dir)

        # 摘要
        elapsed = time.time() - t0
        summarize_outputs(project_root, cfg, cv_acc, elapsed, logger)

    except Exception as e:
        err_log = project_root / "logs" / "error.log"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        with open(err_log, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {repr(e)}\n")
        raise

if __name__ == "__main__":
    main()
