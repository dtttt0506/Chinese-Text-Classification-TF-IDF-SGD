# -*- coding: utf-8 -*-
import re
import jieba
import logging as pylog
jieba.setLogLevel(pylog.ERROR)  # 静音 jieba 前缀词典日志

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
