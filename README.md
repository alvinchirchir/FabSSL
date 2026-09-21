# Label-Efficient Image Classification

Reaching high-accuracy image classification with a fraction of the labels, by combining
self-supervised pre-training, encoder selection, and active learning. Benchmarked on a
demanding real-world setting: semiconductor wafer defect inspection, where expert labels
are slow and expensive to obtain.

---

## The problem

Supervised deep learning needs large labeled datasets. In many real domains the images are
plentiful but labels are the bottleneck: each one needs a domain expert, and the visual
patterns are subtle enough that annotation takes minutes, not seconds. The practical
question is not "what is the best model" but "how do I reach a useful accuracy with the
fewest expert labels."

This project isolates and measures the three levers that control that trade-off, and shows
how they interact as the labeling budget grows.

## The three levers

1. **Self-supervised pre-training.** Learn representations from the unlabeled image archive
   before any labels are spent, using contrastive and self-distillation objectives
   (SimCLR, BYOL, DINOv2), and compare against natural-image transfer (ImageNet) and random
   initialization.
2. **Encoder architecture.** Convolutional (ResNet-18) versus transformer (ViT-S/16)
   backbones, evaluated under identical protocols.
3. **Active learning.** Choose which samples to label next with informed acquisition
   (Diverse Mini-Batch, TypiClust) versus random selection.

Nine encoder configurations (two architectures x five pre-training regimes) are each run
with three acquisition strategies, giving a controlled 27-way comparison per dataset across
a sweep of labeling budgets.

## How it is measured

- **Frozen-encoder linear probe:** the backbone is held fixed and only a linear head is
  trained, so each score reflects representation quality rather than end-to-end fine-tuning.
- **Balanced macro-F1** under heavy class imbalance, with 15-fold cross-validation
  (5-fold for the active-learning runs).
- **Area under the learning curve (AULC):** accuracy integrated over the labeling budget,
  giving a single, budget-aware number to compare acquisition strategies with different
  round counts on a common basis.
- **Statistical testing:** Wilcoxon signed-rank tests (alpha = 0.05) on paired folds, so
  reported differences are significant, not noise.

## What it found

- **Self-supervised pre-training is the dominant lever.** At the smallest labeling budget,
  domain SSL with a ViT-S/16 beat ImageNet transfer by 21.6 points on one dataset and
  5.9 points on the other (p < 0.001). The best configuration reached ~94% balanced accuracy
  on both.
- **Label efficiency, quantified.** The combined pipeline reached 80% F1 with fewer than
  1,000 expert labels, against a transfer-learning baseline that needed several thousand to
  get close.
- **Architecture and pre-training trade places with budget.** At tiny budgets a weaker
  backbone with strong domain pre-training beats a stronger backbone with generic transfer;
  the ordering reverses once labels are plentiful.
- **Active learning helps least when you expect it most.** Informed acquisition gave real
  gains for weak or generic encoders (up to 7.6 AULC points), but the benefit shrank as the
  representation improved and went negative for the strongest encoder, where random selection
  matched or beat it. A useful, counterintuitive result: invest in the representation, not the
  acquisition pipeline.

## What this demonstrates

- Self-supervised and representation learning (contrastive and self-distillation objectives)
- Transfer learning and domain adaptation under distribution shift
- Active learning and label-efficient / data-efficient modeling
- Rigorous evaluation: cross-validation, class-imbalance-aware metrics, paired significance
  testing, budget-aware curve analysis
- Controlled, reproducible benchmarking at scale (dozens of model-strategy combinations)
- Backbone comparison across CNN and vision-transformer families

<!-- ## Stack

PyTorch, with standard vision and ML tooling (torchvision / timm for backbones,
scikit-learn for the linear probe and metrics, NumPy, Matplotlib for analysis). -->

<!-- ## Repository layout

> Fill in with your actual structure, for example:

```
data/            # dataset loaders and splits
encoders/        # backbones and pre-training (SimCLR, BYOL, DINOv2)
active_learning/ # acquisition strategies (random, DMB, TypiClust)
eval/            # linear probe, metrics, cross-validation, statistics
figures/         # analysis and result plots
``` -->

## Getting started

> Add environment and run instructions, for example:

```bash
pip install -r requirements.txt
python -m eval.linear_probe --config configs/vit_simclr.yaml
```
