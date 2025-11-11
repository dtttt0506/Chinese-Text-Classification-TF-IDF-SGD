# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# 我们仍然保留原来的向量器构建（供 grid 使用）
from src.preprocess import build_vectorizer, load_stopwords, tokenizer_factory

# ------------------------------
# 工具：把网格参数做笛卡尔积
# ------------------------------
def _grid_params_iterator(grid: Dict) -> Dict:
    from itertools import product
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in product(*vals):
        d = dict(zip(keys, combo))
        # 统一把 ngram_range 从 list -> tuple（避免 sklearn 参数校验报错）
        for k in list(d.keys()):
            if "ngram_range" in k and isinstance(d[k], list):
                d[k] = tuple(d[k])
        yield d

# ------------------------------
# Grid Search 用的 Pipeline 构建
# ------------------------------
def _build_pipeline(model_name: str, cfg: Dict) -> Pipeline:
    vec = build_vectorizer(cfg)  # 这里仍然使用 TF-IDF + jieba（闭包只在内存中，不参与 pickle 前的保存）
    if model_name == "nb":
        clf = MultinomialNB()
    elif model_name == "svm":
        clf = LinearSVC()
    elif model_name == "lr":
        clf = LogisticRegression(max_iter=4000, n_jobs=-1, solver="saga")
    else:
        raise ValueError("未知模型")
    return Pipeline([("tfidf", vec), ("clf", clf)])

# ------------------------------
# 传统网格搜索（带 tqdm 进度 & 每折指标）
# ------------------------------
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
            # 构建 pipeline 并设置参数（含 ngram_range 的安全转换）
            pipe = _build_pipeline(name, cfg)
            safe_combo = {}
            for k, v in combo.items():
                if "ngram_range" in k and isinstance(v, list):
                    safe_combo[k] = tuple(v)
                else:
                    safe_combo[k] = v
            pipe.set_params(**safe_combo)

            # K 折
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

    # 保存最佳模型（注意：若 build_vectorizer 使用了闭包 tokenizer，这里通常可 pickle；
    # 如仍遇到 pickle 问题，建议将 grid 路径也改为“预分词+str.split”的策略。）
    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ckpt_dir / "best_pipeline.joblib")
    logger.info(f"[保存] 最优模型 {best_model_name} | cv_acc={best_score:.4f} | params={best_param}")
    return best_pipeline, best_model_name, best_score

# ==========================================================
#   SGD 路径（修改一）：可序列化的分词器 + 一次性预分词加速
# ==========================================================

class JiebaCutTransformer(BaseEstimator, TransformerMixin):
    """
    一个可序列化（可 pickle）的分词 Transformer。
    - transform 返回“以空格分隔”的 token 串，供 TfidfVectorizer(tokenizer=str.split) 使用
    - 这样整个 Pipeline 在推理时可以接收“原始中文文本”，先切词再向量化
    """
    def __init__(self, stopwords_path: str = "config/cn_stopwords.txt"):
        self.stopwords_path = stopwords_path
        self._stop = None  # 运行时加载

    def fit(self, X, y=None):
        # 延迟加载停用词
        if self._stop is None:
            self._stop = load_stopwords(self.stopwords_path)
        return self

    def transform(self, X):
        if self._stop is None:
            self._stop = load_stopwords(self.stopwords_path)
        tok = tokenizer_factory(self._stop)  # 这是闭包，但仅在 transform 内使用，不进入持久化状态
        out = []
        for s in X:
            # 直接切词 -> 空格拼接
            out.append(" ".join(tok(s)))
        return out

def run_sgd_training(
    X_train: List[str], y_train: List[str],
    X_val: List[str], y_val: List[str],
    cfg: Dict, project_root: Path,
    vectorizer,            # 保持签名兼容（此参数在本实现中不再使用）
    label_names: List[str], logger, timer
):
    """
    修改一的目标：
    - 训练阶段：一次性调用 jieba 分词（避免每 batch 重复分词），并用 str.split 的 TF-IDF。
    - 推理阶段：保存为 Pipeline([jieba, tfidf, clf])，因此对“原始中文文本”也能直接 predict。
    - 避免闭包进 Pipeline，彻底修复 joblib PicklingError。
    """
    # ---------------------------
    # 1) 一次性预分词（训练/验证）
    # ---------------------------
    logger.info("[SGD] 预分词 train/val ...")
    jieba_tr = JiebaCutTransformer(stopwords_path=cfg["paths"].get("stopwords", "config/cn_stopwords.txt"))
    # 先 fit 一下（加载停用词）
    jieba_tr.fit(None)

    # 用 tqdm 包裹手动预分词（更快且有进度条）
    def _pretokenize_once(texts: List[str], desc: str):
        out = []
        tok = tokenizer_factory(jieba_tr._stop)
        for s in tqdm(texts, desc=desc, mininterval=cfg["viz"].get("tqdm_mininterval", 0.2)):
            out.append(" ".join(tok(s)))
        return out

    Xtr_tok = _pretokenize_once(X_train, desc="预分词(train)")
    Xva_tok = _pretokenize_once(X_val,   desc="预分词(val)")

    # ---------------------------
    # 2) TF-IDF（str.split）向量化
    # ---------------------------
    use_bigram = cfg["preprocess"].get("use_bigram", True)
    max_features = cfg["preprocess"].get("max_features", 50000)
    min_df = cfg["preprocess"].get("min_df", 2)
    tfidf = TfidfVectorizer(
        tokenizer=str.split,       # 关键：避免闭包进入 Pipeline
        token_pattern=None,
        preprocessor=None,
        ngram_range=(1,2) if use_bigram else (1,1),
        max_features=max_features,
        min_df=min_df
    )

    logger.info("[SGD] 拟合向量器...")
    Xtr = tfidf.fit_transform(Xtr_tok)
    Xva = tfidf.transform(Xva_tok)

    # ---------------------------
    # 3) SGD 增量训练（epoch 级实时指标）
    # ---------------------------
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

    # ---------------------------
    # 4) 组装可序列化 Pipeline 并保存
    #    Pipeline 接收原始中文文本：先 jieba 切词 -> 空格串 -> TF-IDF -> 分类
    # ---------------------------
    best_pipeline = Pipeline([
        ("jieba", jieba_tr),  # 可序列化的分词器
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ckpt_dir / "best_pipeline.joblib")
    logger.info(f"[保存] SGD 最终模型 -> {ckpt_dir/'best_pipeline.joblib'}")

    # 返回验证集精度用于运行摘要（用“原始文本”走 Pipeline 预测，验证端到端一致性）
    y_pred = best_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return best_pipeline, "sgd", float(acc)

def run_fast_mode(
    X_train: List[str], y_train: List[str],
    X_val: List[str], y_val: List[str],
    cfg: Dict, project_root: Path,
    label_names: List[str], logger, timer
):
    """
    FAST 模式：HashingVectorizer + TfidfTransformer + LinearSVC
    - 先把原始中文通过 JiebaCutTransformer 切词为“空格分词文本”
    - HashingVectorizer 无需拟合，极快；TfidfTransformer 拟合 IDF
    - 输出的 Pipeline 含 jieba -> hashing -> tfidf -> clf，可直接对原文 predict
    """
    # 若没有显式 val 集，则从训练集中切分一部分做验证
    if len(X_val) == 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=cfg.get("val_ratio", 0.1),
            random_state=42, stratify=y_train
        )

    # 1) 分词（可序列化的 Transformer）
    jieba_tr = JiebaCutTransformer(stopwords_path=cfg["paths"].get("stopwords", "config/cn_stopwords.txt"))

    # 2) 特征与模型
    n_features = int(cfg.get("fast", {}).get("n_features", 2**20))
    use_bigram = bool(cfg.get("fast", {}).get("use_bigram", True))
    C = float(cfg.get("fast", {}).get("C", 1.0))

    hash_vec = HashingVectorizer(
        tokenizer=str.split, token_pattern=None, analyzer="word",
        n_features=n_features, alternate_sign=False,
        ngram_range=(1, 2) if use_bigram else (1, 1)
    )
    tfidf_tr = TfidfTransformer()
    clf = LinearSVC(C=C)

    # 3) 组装端到端 Pipeline：原文 -> jieba -> hashing -> tfidf -> clf
    pipe = Pipeline([
        ("jieba", jieba_tr),
        ("hash", hash_vec),
        ("tfidf", tfidf_tr),
        ("clf", clf)
    ])

    logger.info("[FAST] 拟合 LinearSVC ...")
    pipe.fit(X_train, y_train)

    # 4) 验证集评估
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    logger.info(f"[FAST] val_acc={acc:.4f} val_f1={f1:.4f}")

    # 5) 保存
    ckpt_dir = project_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ckpt_dir / "best_pipeline.joblib")
    logger.info(f"[保存] FAST 模型 -> {ckpt_dir/'best_pipeline.joblib'}")

    # 返回验证集精度作为摘要
    return pipe, "fast_hashing_svm", float(acc)
