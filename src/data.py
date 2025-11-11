# -*- coding: utf-8 -*-
import os, re
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def _clean_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _scan_dir_texts(dir_path: Path, desc: str) -> Tuple[List[str], List[str], List[str]]:
    texts, labels, files = [], [], []
    if not dir_path.exists():
        return texts, labels, files
    subdirs = [p for p in dir_path.glob("*") if p.is_dir()]
    for sd in tqdm(subdirs, desc=f"扫描{desc}类别", mininterval=0.2):
        label = sd.name
        for fp in tqdm(list(sd.glob("*.txt")), desc=f"{label}", leave=False, mininterval=0.2):
            try:
                content = fp.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = fp.read_text(encoding="gbk", errors="ignore")
            content = _clean_text(content)
            if len(content) >= 5:
                texts.append(content); labels.append(label); files.append(str(fp))
    return texts, labels, files

def load_data_sets(project_root: Path, cfg: dict):
    paths = cfg["paths"]
    train_dir = project_root / paths["train_dir"]
    val_dir = project_root / paths["val_dir"]
    test_dir = project_root / paths["test_dir"]
    use_val_dir = cfg.get("use_val_dir", True)
    val_ratio = cfg.get("val_ratio", 0.1)

    X_train, y_train, _ = _scan_dir_texts(train_dir, "训练")
    if use_val_dir:
        X_val, y_val, _ = _scan_dir_texts(val_dir, "验证")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
        )
    X_test, y_test, _ = _scan_dir_texts(test_dir, "测试")

    label_names = sorted(list(set(y_train + y_val + y_test)))
    print(f"[数据] 训练:{len(X_train)} 验证:{len(X_val)} 测试:{len(X_test)} 类别数:{len(label_names)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_names
