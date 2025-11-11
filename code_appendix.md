# 论文代码附录（主要执行代码）

本附录汇集项目的主要执行代码，以便论文读者快速理解实现细节。内容按模块组织，保持与仓库一致的结构与函数接口。

> 说明：代码为当前实现的关键片段，涵盖数据加载、预处理（分词与向量化）、三种训练模式、评估与预测/交互使用。为便于阅读，不包含与论文无关的辅助打印或边缘处理。

---

## 1. 数据加载（`src/data.py`）
```python
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
```

---

## 2. 预处理与向量化（`src/preprocess.py`）
```python
# -*- coding: utf-8 -*-
import re
import jieba
import logging as pylog
jieba.setLogLevel(pylog.ERROR)

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_STOPWORDS = {
    "的","了","和","是","就","都","而","及","与","着","或","一个",
    "没有","我们","你们","他们","她们","是否","对于","以及","如果",
    "因为","所以","但是","而且","可以","因此","并且","同时","通过",
    "非常","比较","这个","那个","这些","那些"
}


def load_stopwords(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}
    except Exception:
        return DEFAULT_STOPWORDS


def tokenizer_factory(stopwords: set):
    def tok(text: str):
        out = []
        for t in jieba.cut(text):
            t = t.strip()
            if not t or t in stopwords:
                continue
            if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", t):
                continue
            out.append(t)
        return out
    return tok


def build_vectorizer(cfg: dict):
    sw = load_stopwords(cfg["paths"].get("stopwords", "config/cn_stopwords.txt"))
    use_bigram = cfg["preprocess"].get("use_bigram", True)
    max_features = cfg["preprocess"].get("max_features", 50000)
    min_df = cfg["preprocess"].get("min_df", 2)
    return TfidfVectorizer(
        tokenizer=tokenizer_factory(sw),
        token_pattern=None,
        ngram_range=(1,2) if use_bigram else (1,1),
        max_features=max_features,
        min_df=min_df
    )
```

---

## 3. 训练器：网格搜索（`src/trainers.py`）
```python
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import build_vectorizer


def _grid_params_iterator(grid: Dict) -> Dict:
    from itertools import product
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in product(*vals):
        d = dict(zip(keys, combo))
        for k in list(d.keys()):
            if "ngram_range" in k and isinstance(d[k], list):
                d[k] = tuple(d[k])
        yield d


def _build_pipeline(model_name: str, cfg: Dict) -> Pipeline:
    vec = build_vectorizer(cfg)
    if model_name == "nb":
        clf = MultinomialNB()
    elif model_name == "svm":
        clf = LinearSVC()
    elif model_name == "lr":
        clf = LogisticRegression(max_iter=4000, n_jobs=-1, solver="saga")
    else:
        raise ValueError("未知模型")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def run_grid_search(
    X: List[str], y: List[str], cfg: Dict, project_root: Path,
    label_names: List[str], logger, timer
):
    params_all = cfg["grid_params"]
    cv_folds = cfg["train"]["cv_folds"]

    best_score, best_model_name, best_pipeline, best_param = -1, None, None, None
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, grid in params_all.items():
        logger.info(f"\n[网格搜索] 模型: {name}")
        combos = list(_grid_params_iterator(grid))
        pbar = tqdm(combos, desc=f"Grid-{name}", mininterval=cfg["viz"].get("tqdm_mininterval", 0.2))
        for combo in pbar:
            pipe = _build_pipeline(name, cfg)
            safe_combo = {}
            for k, v in combo.items():
                if "ngram_range" in k and isinstance(v, list):
                    safe_combo[k] = tuple(v)
                else:
                    safe_combo[k] = v
            pipe.set_params(**safe_combo)

            fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
                X_tr = [X[i] for i in tr_idx]; y_tr = [y[i] for i in tr_idx]
                X_va = [X[i] for i in va_idx]; y_va = [y[i] for i in va_idx]
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_va)
                acc = accuracy_score(y_va, y_pred)
                f1 = f1_score(y_va, y_pred, average="macro")
                logger.info(f"[{name}] params={safe_combo} fold={fold}/{cv_folds} acc={acc:.4f} f1={f1:.4f}")
                fold_scores.append(acc)

            cv_acc = float(np.mean(fold_scores))
            pbar.set_postfix({"cv_acc": f"{cv_acc:.4f}"})
            if cv_acc > best_score:
                best_score, best_model_name, best_pipeline, best_param = cv_acc, name, pipe, safe_combo

    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ckpt_dir / "best_pipeline.joblib")
    logger.info(f"[保存] 最优模型 {best_model_name} | cv_acc={best_score:.4f} | params={best_param}")
    return best_pipeline, best_model_name, best_score
```

---

## 4. 训练器：可序列化分词与 SGD 增量训练（`src/trainers.py`）
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from src.preprocess import load_stopwords, tokenizer_factory


class JiebaCutTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords_path: str = "config/cn_stopwords.txt"):
        self.stopwords_path = stopwords_path
        self._stop = None
    def fit(self, X, y=None):
        if self._stop is None:
            self._stop = load_stopwords(self.stopwords_path)
        return self
    def transform(self, X):
        if self._stop is None:
            self._stop = load_stopwords(self.stopwords_path)
        tok = tokenizer_factory(self._stop)
        out = [" ".join(tok(s)) for s in X]
        return out


def run_sgd_training(
    X_train, y_train, X_val, y_val,
    cfg, project_root, vectorizer, label_names, logger, timer
):
    jieba_tr = JiebaCutTransformer(stopwords_path=cfg["paths"].get("stopwords", "config/cn_stopwords.txt"))
    jieba_tr.fit(None)

    def _pretokenize_once(texts, desc: str):
        from tqdm import tqdm
        tok = tokenizer_factory(jieba_tr._stop)
        return [" ".join(tok(s)) for s in tqdm(texts, desc=desc, mininterval=cfg["viz"].get("tqdm_mininterval", 0.2))]

    Xtr_tok = _pretokenize_once(X_train, desc="预分词(train)")
    Xva_tok = _pretokenize_once(X_val,   desc="预分词(val)")

    tfidf = TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, preprocessor=None,
        ngram_range=(1,2) if cfg["preprocess"].get("use_bigram", True) else (1,1),
        max_features=cfg["preprocess"].get("max_features", 50000),
        min_df=cfg["preprocess"].get("min_df", 2)
    )
    Xtr = tfidf.fit_transform(Xtr_tok)
    Xva = tfidf.transform(Xva_tok)

    clf = SGDClassifier(
        loss=cfg["train"]["sgd"].get("loss", "log_loss"),
        learning_rate="optimal",
        alpha=cfg["train"]["sgd"].get("learning_rate", 5e-4),
        random_state=42
    )

    classes = np.unique(y_train)
    batch_size = cfg["train"]["sgd"].get("batch_size", 256)
    max_epochs = cfg["train"]["sgd"].get("max_epochs", 5)
    eval_every = cfg["train"]["sgd"].get("eval_every", 1)

    idx = np.arange(Xtr.shape[0])
    from tqdm import tqdm
    pbar = tqdm(range(1, max_epochs + 1), desc="SGD-epochs", mininterval=cfg["viz"].get("tqdm_mininterval", 0.2))
    for epoch in pbar:
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            sel = idx[start:start+batch_size]
            clf.partial_fit(Xtr[sel], np.array(y_train)[sel], classes=classes)

        if epoch % eval_every == 0:
            y_pred = clf.predict(Xva)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")
            pbar.set_postfix({"val_acc": f"{acc:.4f}", "val_f1": f"{f1:.4f}"})
            logger.info(f"[SGD] epoch={epoch}/{max_epochs} val_acc={acc:.4f} val_f1={f1:.4f}")

    best_pipeline = Pipeline([
        ("jieba", jieba_tr),
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ckpt_dir / "best_pipeline.joblib")

    y_pred = best_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return best_pipeline, "sgd", float(acc)
```

---

## 5. 训练器：FAST 高效模式（`src/trainers.py`）
```python
from pathlib import Path
from typing import List
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# 依赖可序列化的 JiebaCutTransformer（见上文）

def run_fast_mode(
    X_train: List[str], y_train: List[str],
    X_val: List[str], y_val: List[str],
    cfg: dict, project_root: Path,
    label_names: List[str], logger, timer
):
    if len(X_val) == 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=cfg.get("val_ratio", 0.1),
            random_state=42, stratify=y_train
        )

    jieba_tr = JiebaCutTransformer(stopwords_path=cfg["paths"].get("stopwords", "config/cn_stopwords.txt"))

    n_features = int(cfg.get("fast", {}).get("n_features", 2**20))
    use_bigram = bool(cfg.get("fast", {}).get("use_bigram", True))

    hash_vec = HashingVectorizer(
        tokenizer=str.split, token_pattern=None, analyzer="word",
        n_features=n_features, alternate_sign=False,
        ngram_range=(1, 2) if use_bigram else (1, 1)
    )
    tfidf_tr = TfidfTransformer()
    clf = LinearSVC(C=float(cfg.get("fast", {}).get("C", 1.0)))

    pipe = Pipeline([
        ("jieba", jieba_tr),
        ("hash", hash_vec),
        ("tfidf", tfidf_tr),
        ("clf", clf)
    ])

    logger.info("[FAST] 拟合 LinearSVC ...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    logger.info(f"[FAST] val_acc={acc:.4f} val_f1={f1:.4f}")

    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ckpt_dir / "best_pipeline.joblib")

    return pipe, "fast_hashing_svm", float(acc)
```

---

## 6. 评估与可视化（`src/evaluate.py`）
```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_and_plot(model, X_test, y_test, label_names, out_dir: Path, save_confmat=True, logger=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=4)
    (out_dir / "report.txt").write_text(rep, encoding="utf-8")

    if logger:
        logger.info("\n=== 测试集评估 ===")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("\n" + rep)

    if save_confmat:
        cm = confusion_matrix(y_test, y_pred, labels=label_names)
        plt.figure(figsize=(max(6, len(label_names)*0.6), max(5, len(label_names)*0.6)))
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick = np.arange(len(label_names))
        plt.xticks(tick, label_names, rotation=45, ha="right")
        plt.yticks(tick, label_names)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = int(cm[i,j])
                plt.text(j, i, str(v), ha="center", va="center",
                         color="white" if v > thresh else "black")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        fig_path = out_dir / "confusion_matrix.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        if logger:
            logger.info(f"[图表] 混淆矩阵已保存：{fig_path}")
```

---

## 7. 预测与标签空间（`src/predict.py`）
```python
from pathlib import Path
import joblib


def save_label_space(model, ckpt_dir: Path):
    try:
        classes_ = getattr(model.named_steps["clf"], "classes_", None)
        if classes_ is not None:
            (ckpt_dir / "meta.txt").write_text("classes:\n" + "\n".join(map(str, classes_)), encoding="utf-8")
    except Exception:
        pass


def load_pipeline(ckpt_dir: Path):
    return joblib.load(ckpt_dir / "best_pipeline.joblib")


def predict_texts(model, texts):
    return model.predict(texts)
```

---

## 8. 交互式预测脚本（`interactive_predict.py`）
```python
import argparse, pathlib, sys, time, json, numpy as np, joblib

# 省略：_softmax 与 _get_class_probs（基于 predict_proba 或 decision_function 回退）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best_pipeline.joblib")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"[错误] 找不到模型文件：{model_path}")
        sys.exit(1)

    pipe = joblib.load(model_path)
    classes = getattr(pipe, "classes_", None) or getattr(pipe.named_steps.get("clf"), "classes_", None)

    writer = None
    if args.save:
        out_dir = pathlib.Path("output"); out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"predictions_{stamp}.jsonl"
        writer = out_file.open("w", encoding="utf-8")

    while True:
        try:
            text = input("\n请输入待分类文本（exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q", "q"}:
            break
        if text == "/help":
            print("指令：exit/quit/:q/q 退出；/help 帮助")
            continue

        try:
            pred = pipe.predict([text])[0]
            print(f"→ 预测类别：{pred}")
        except Exception as e:
            print(f"[错误] 预测失败：{e}")
            continue

        if writer is not None:
            rec = {"text": text, "pred": str(pred)}
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.flush()

    if writer is not None:
        writer.close()
```

---

## 9. 主运行入口（`run.py`）片段
```python
import yaml
from pathlib import Path
from src.utils import set_global_seed, setup_dirs, setup_logger, RunnerTimer, summarize_outputs
from src.data import load_data_sets
from src.preprocess import build_vectorizer
from src.trainers import run_grid_search, run_sgd_training, run_fast_mode
from src.evaluate import evaluate_and_plot
from src.predict import save_label_space


def main():
    project_root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((project_root / "config" / "config.yaml").read_text(encoding="utf-8"))
    set_global_seed(cfg.get("seed", 42))
    paths = cfg["paths"]
    setup_dirs(project_root, paths)
    logger = setup_logger(project_root / paths["logs_dir"])
    timer = RunnerTimer()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_names = load_data_sets(project_root, cfg)

    mode = cfg["train"]["mode"]
    if mode == "grid":
        best_model, best_name, cv_acc = run_grid_search(X_train, y_train, cfg, project_root, label_names, logger, timer)
    elif mode == "sgd":
        vectorizer = build_vectorizer(cfg)
        best_model, best_name, cv_acc = run_sgd_training(X_train, y_train, X_val, y_val, cfg, project_root, vectorizer, label_names, logger, timer)
    elif mode == "fast":
        best_model, best_name, cv_acc = run_fast_mode(X_train, y_train, X_val, y_val, cfg, project_root, label_names, logger, timer)
    else:
        raise ValueError("train.mode 仅支持 'fast' | 'sgd' | 'grid'")

    out_dir = project_root / paths["output_dir"]
    ckpt_dir = project_root / paths["checkpoints_dir"]
    evaluate_and_plot(best_model, X_test, y_test, label_names, out_dir=out_dir, save_confmat=cfg["viz"].get("save_confusion_matrix", True), logger=logger)
    save_label_space(best_model, ckpt_dir)
```

---

以上片段即为论文所需的主要执行代码，保持与仓库一致。若需要我将此文件转换为 LaTeX `listings` 格式或进一步精简，请告知具体要求。