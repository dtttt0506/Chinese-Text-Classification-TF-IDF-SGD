# -*- coding: utf-8 -*-
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
