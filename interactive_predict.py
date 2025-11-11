# -*- coding: utf-8 -*-
"""
交互式文本分类推理脚本
- 从 checkpoints/best_pipeline.joblib 加载已训练的 Pipeline
- 支持 Top-K 概率展示（若管道支持 predict_proba；否则用 softmax(decision_function) 近似）
- 支持 --model / --topk / --save 三个参数
"""

import argparse
import pathlib
import sys
import time
import json
import numpy as np
import joblib

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / np.sum(e)
    return s

def _get_class_probs(model, text: str):
    """返回 (labels, probs)，若无法得到概率则返回 (labels, None)"""
    # 优先用 predict_proba
    try:
        proba = model.predict_proba([text])[0]
        labels = getattr(model, "classes_", None)
        if labels is None:
            # 对于Pipeline：clf在named_steps中
            clf = getattr(model, "named_steps", {}).get("clf", None)
            labels = getattr(clf, "classes_", None)
        return labels, proba
    except Exception:
        pass

    # 回退：decision_function → softmax 近似
    try:
        scores = model.decision_function([text])[0]
        # 二分类时 scores 可能是标量或长度为2；统一本为1D
        scores = np.atleast_1d(scores)
        if scores.ndim == 0:
            scores = np.array([ -scores, scores ])  # 尝试构造两类打分
        probs = _softmax(scores)
        labels = getattr(model, "classes_", None)
        if labels is None:
            clf = getattr(model, "named_steps", {}).get("clf", None)
            labels = getattr(clf, "classes_", None)
        return labels, probs
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best_pipeline.joblib",
                        help="模型路径（相对或绝对），默认 checkpoints/best_pipeline.joblib")
    parser.add_argument("--topk", type=int, default=3, help="显示Top-K类别（若可用）")
    parser.add_argument("--save", action="store_true", help="是否将交互记录保存到 output/predictions_*.jsonl")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"[错误] 找不到模型文件：{model_path}")
        print("请确认已训练完成且文件位于 checkpoints/best_pipeline.joblib 或使用 --model 指定路径。")
        sys.exit(1)

    print(f"[加载模型] {model_path}")
    pipe = joblib.load(model_path)

    # 读取类别名（若可用）
    classes = getattr(pipe, "classes_", None)
    if classes is None and hasattr(pipe, "named_steps"):
        clf = pipe.named_steps.get("clf")
        if clf is not None:
            classes = getattr(clf, "classes_", None)

    if classes is not None:
        print(f"[类别数] {len(classes)}")
        print("[提示] 可输入 exit 退出；可输入 /help 查看帮助。")
    else:
        print("[提示] 未能从模型中读取类别名。仍可正常预测。输入 exit 退出；/help 查看帮助。")

    # 准备保存
    writer = None
    if args.save:
        out_dir = pathlib.Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"predictions_{stamp}.jsonl"
        writer = out_file.open("w", encoding="utf-8")
        print(f"[记录] 交互预测将写入：{out_file}")

    # 交互循环
    while True:
        try:
            text = input("\n请输入待分类文本（exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[结束] Bye.")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q", "q"}:
            print("[结束] Bye.")
            break
        if text == "/help":
            print("指令：\n  exit / quit / :q / q  退出\n  /help  查看帮助\n  普通输入视为待分类文本")
            continue

        # 预测
        try:
            pred = pipe.predict([text])[0]
        except Exception as e:
            print(f"[错误] 预测失败：{e}")
            continue

        print(f"→ 预测类别：{pred}")

        # 打印Top-K概率（若可用）
        labels, probs = _get_class_probs(pipe, text)
        if probs is not None and labels is not None:
            k = max(1, min(args.topk, len(labels)))
            order = np.argsort(probs)[::-1][:k]
            print("→ Top-{}:".format(k))
            for idx in order:
                print(f"   {labels[idx]}: {probs[idx]:.4f}")
        else:
            print("→ 概率分布不可用（模型未提供 predict_proba，或不支持 decision_function 回退）。")

        # 保存一条记录
        if writer is not None:
            rec = {"text": text, "pred": str(pred)}
            if probs is not None and labels is not None:
                rec["topk"] = [
                    {"label": str(labels[i]), "prob": float(probs[i])}
                    for i in np.argsort(probs)[::-1][:max(1, min(args.topk, len(labels)))]
                ]
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.flush()

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    # 兼容 Windows 控制台中文
    if sys.platform.startswith("win"):
        try:
            import msvcrt  # noqa: F401
            import os
            os.system("")  # 触发控制台编码初始化
        except Exception:
            pass
    main()


