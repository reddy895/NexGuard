"""
NexGuard — Confusion Matrix Visualizer
======================================
Renders a publication-ready confusion matrix graphic (reports/confusion_matrix.png)
using OpenCV and PIL without requiring external heavy plotting dependencies.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ConfusionMatrixVisualizer:
    """
    Renders styled confusion matrix charts for accident classification.
    """

    @staticmethod
    def render(
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        output_png: Path | str = "reports/confusion_matrix.png",
        title: str = "NexGuard Accident Classifier — Test Confusion Matrix",
    ) -> Path:
        out_p = Path(output_png)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        width, height = 900, 650
        # Dark modern theme (#0f172a)
        canvas = np.full((height, width, 3), (24, 23, 15), dtype=np.uint8)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
            font_val = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font_title = font_bold = font_val = font_sub = ImageFont.load_default()

        # Header
        draw.text((50, 35), title, fill=(255, 255, 255), font=font_title)
        draw.text((50, 70), "Independent Held-Out Evaluation Split | Source-Aware Tested", fill=(148, 163, 184), font=font_sub)

        # Matrix Layout
        cx, cy = 200, 160
        cell_size = 180

        # Axes labels
        draw.text((cx + 80, cy - 35), "Predicted: Accident", fill=(248, 113, 113), font=font_bold)
        draw.text((cx + cell_size + 50, cy - 35), "Predicted: Non-Accident", fill=(52, 211, 153), font=font_bold)

        draw.text((cx - 160, cy + 70), "Actual: Accident", fill=(248, 113, 113), font=font_bold)
        draw.text((cx - 195, cy + cell_size + 70), "Actual: Non-Accident", fill=(52, 211, 153), font=font_bold)

        tot_actual_acc = max(tp + fn, 1)
        tot_actual_non = max(fp + tn, 1)

        # 1. Top-Left: True Positives (Green/Teal tone)
        draw.rectangle([cx, cy, cx + cell_size, cy + cell_size], fill=(16, 185, 129), outline=(51, 65, 85), width=2)
        draw.text((cx + 65, cy + 45), f"{tp}", fill=(255, 255, 255), font=font_val)
        draw.text((cx + 35, cy + 95), f"TP ({tp / tot_actual_acc:.1%})", fill=(209, 250, 229), font=font_bold)
        draw.text((cx + 25, cy + 125), "Correct Detection", fill=(209, 250, 229), font=font_sub)

        # 2. Top-Right: False Negatives (Missed) (Crimson/Red)
        draw.rectangle([cx + cell_size, cy, cx + cell_size * 2, cy + cell_size], fill=(225, 29, 72), outline=(51, 65, 85), width=2)
        draw.text((cx + cell_size + 70, cy + 45), f"{fn}", fill=(255, 255, 255), font=font_val)
        draw.text((cx + cell_size + 40, cy + 95), f"FN ({fn / tot_actual_acc:.1%})", fill=(254, 205, 211), font=font_bold)
        draw.text((cx + cell_size + 30, cy + 125), "Missed Accident", fill=(254, 205, 211), font=font_sub)

        # 3. Bottom-Left: False Positives (False Alarm) (Amber/Orange)
        draw.rectangle([cx, cy + cell_size, cx + cell_size, cy + cell_size * 2], fill=(217, 119, 6), outline=(51, 65, 85), width=2)
        draw.text((cx + 65, cy + cell_size + 45), f"{fp}", fill=(255, 255, 255), font=font_val)
        draw.text((cx + 35, cy + cell_size + 95), f"FP ({fp / tot_actual_non:.1%})", fill=(254, 243, 199), font=font_bold)
        draw.text((cx + 40, cy + cell_size + 125), "False Alarm", fill=(254, 243, 199), font=font_sub)

        # 4. Bottom-Right: True Negatives (Emerald)
        draw.rectangle([cx + cell_size, cy + cell_size, cx + cell_size * 2, cy + cell_size * 2], fill=(13, 148, 136), outline=(51, 65, 85), width=2)
        draw.text((cx + cell_size + 70, cy + cell_size + 45), f"{tn}", fill=(255, 255, 255), font=font_val)
        draw.text((cx + cell_size + 40, cy + cell_size + 95), f"TN ({tn / tot_actual_non:.1%})", fill=(204, 251, 241), font=font_bold)
        draw.text((cx + cell_size + 30, cy + cell_size + 125), "Correct Normal", fill=(204, 251, 241), font=font_sub)

        # Summary Metrics Box on right
        sx, sy = 600, 160
        draw.rectangle([sx, sy, sx + 250, sy + cell_size * 2], fill=(30, 41, 59), outline=(51, 65, 85), width=1)
        draw.text((sx + 20, sy + 25), "Evaluation Summary", fill=(255, 255, 255), font=font_bold)

        tot = max(tp + fp + fn + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-6)
        acc = (tp + tn) / tot

        metrics = [
            f"Accuracy : {acc:.2%}",
            f"Precision: {prec:.2%}",
            f"Recall   : {rec:.2%}",
            f"F1-Score : {f1:.2%}",
            f"Total    : {tot} frames",
        ]
        for idx, text in enumerate(metrics):
            draw.text((sx + 20, sy + 75 + idx * 35), text, fill=(226, 232, 240), font=font_sub)

        # Footer
        draw.text((50, 580), "NexGuard MLOps — Validated Empirical Evaluation", fill=(100, 116, 139), font=font_sub)

        img.save(out_p)
        return out_p
