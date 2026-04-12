"""
Local dataset generator — produces the same output format as generate_dataset.py
without requiring an OpenAI API key. Use this for testing the fine-tuning pipeline.
"""
import json
import os
import random

OUTPUT_DIR = "data"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

PAIRS: list[dict] = [
    # linear regression
    {
        "prompt": "Can you explain how linear regression works and when it is appropriate to use it?",
        "response": (
            "Linear regression models the relationship between one or more input features and a continuous output by fitting "
            "a line (or hyperplane) through the data: y = w₀ + w₁x₁ + … + wₙxₙ. Weights are learned by minimizing Mean "
            "Squared Error (MSE) using Ordinary Least Squares or gradient descent.\n\n"
            "The algorithm assumes linearity, independence of residuals, homoscedasticity, and normality of errors. When "
            "these hold, linear regression is the Best Linear Unbiased Estimator (BLUE). Multicollinearity can be detected "
            "via the Variance Inflation Factor (VIF) and addressed with Ridge or Lasso regularization.\n\n"
            "Linear regression is best when the target is continuous, the relationship with features is approximately "
            "linear, and interpretability matters. Common uses include house price prediction and sales forecasting. "
            "Regularized variants (Ridge/Lasso) prevent overfitting in high-dimensional settings."
        ),
    },
    {
        "prompt": "What is the difference between Ridge and Lasso regression?",
        "response": (
            "Ridge (L2) adds λ∑wᵢ² to the loss, shrinking all coefficients toward zero but never exactly to zero. "
            "Lasso (L1) adds λ∑|wᵢ|, which can drive coefficients to exactly zero, performing automatic feature selection.\n\n"
            "Geometrically, the L2 constraint is a sphere (smooth, no corners) so the optimum rarely lands on an axis. "
            "The L1 constraint is a diamond with corners on the axes, making it likely that weights become exactly zero — "
            "producing sparse models.\n\n"
            "Prefer Ridge when most features are relevant and should be retained but shrunk. Prefer Lasso when only a "
            "subset of features matters and you need an interpretable sparse model. Elastic Net combines both penalties "
            "and handles groups of correlated features better than Lasso alone."
        ),
    },
    # logistic regression
    {
        "prompt": "How does logistic regression perform classification despite having 'regression' in its name?",
        "response": (
            "Logistic regression extends linear regression by passing the linear combination through the sigmoid function "
            "σ(z) = 1/(1+e^(-z)), mapping any real number to a probability in (0,1). This probability is then thresholded "
            "(usually at 0.5) to produce a class label.\n\n"
            "Weights are learned by maximizing the log-likelihood, equivalent to minimizing Binary Cross-Entropy: "
            "L = -[y·log(p) + (1-y)·log(1-p)]. This loss is convex, guaranteeing a global minimum with gradient descent. "
            "Each coefficient represents the change in log-odds per unit increase in the feature, making the model interpretable.\n\n"
            "For multi-class problems the Softmax function generalizes logistic regression. It is widely used in medicine, "
            "finance, and NLP because it is fast, interpretable, and provides well-calibrated probabilities. Its main "
            "limitation is the assumption of a linear decision boundary."
        ),
    },
    # decision trees
    {
        "prompt": "How does a decision tree choose the best feature to split on at each node?",
        "response": (
            "At each node, a decision tree evaluates every feature and threshold, selecting the split that best reduces "
            "impurity. For classification, the two most common criteria are Gini Impurity (G = 1 - ∑pᵢ²) and Information "
            "Gain (reduction in Shannon entropy after the split). For regression, variance reduction (MSE decrease) is used.\n\n"
            "The algorithm sorts continuous features and evaluates thresholds between consecutive distinct values — O(n log n) "
            "per feature. CART uses binary splits with Gini; ID3/C4.5 use entropy. The split that maximizes the chosen "
            "criterion is selected recursively until stopping conditions (max depth, min samples) are met.\n\n"
            "Decision trees are prone to overfitting when grown deep and are sensitive to small data changes. Pruning "
            "controls complexity. Despite individual fragility, trees are the building blocks of powerful ensembles — "
            "Random Forests and Gradient Boosting — which combine many trees to achieve strong generalization."
        ),
    },
    # random forests
    {
        "prompt": "What is the intuition behind Random Forests and why are they more robust than a single decision tree?",
        "response": (
            "Random Forests train many decision trees on bootstrapped subsets of training data (bagging) and aggregate "
            "their predictions via majority vote (classification) or averaging (regression). The key innovation is feature "
            "randomness: at each split, only a random subset of features (√p for classification) is considered, "
            "decorrelating the trees.\n\n"
            "The bias-variance tradeoff explains the improvement: a single deep tree has high variance — it changes "
            "dramatically with small data perturbations. Averaging many decorrelated trees reduces variance without "
            "increasing bias, following the law of large numbers for estimator ensembles.\n\n"
            "Random Forests provide out-of-bag (OOB) error estimates free of charge — each tree is evaluated on the ~37% "
            "of samples not in its bootstrap. They also compute feature importance via mean impurity decrease, are robust "
            "to outliers, require little hyperparameter tuning, and handle mixed feature types well."
        ),
    },
    # gradient boosting
    {
        "prompt": "How does gradient boosting differ from Random Forests and what makes it so effective?",
        "response": (
            "While Random Forests build trees in parallel and average them, Gradient Boosting builds trees sequentially: "
            "each new shallow tree corrects the errors of the current ensemble by fitting the negative gradient of the loss "
            "function (residuals for MSE). A learning rate η scales each tree's contribution, and the process repeats.\n\n"
            "This gradient-descent-in-function-space framework generalizes to any differentiable loss. Modern implementations "
            "(XGBoost, LightGBM, CatBoost) add second-order gradient approximations, L1/L2 regularization, histogram-based "
            "split finding, and GPU support, making them dramatically faster than naive boosting.\n\n"
            "Gradient boosting typically outperforms Random Forests on tabular data by focusing capacity on hard examples. "
            "However, it requires careful tuning of learning rate, n_estimators, and max_depth. Early stopping — halting "
            "when validation loss stops improving — is essential to prevent overfitting."
        ),
    },
    # artificial neural networks
    {
        "prompt": "Can you explain how backpropagation works in neural networks?",
        "response": (
            "Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule "
            "in a reverse pass through the computation graph. The forward pass caches activations at each layer. The "
            "backward pass starts at the loss, propagates the gradient layer by layer: "
            "∂L/∂wᵢⱼ = ∂L/∂aⱼ · ∂aⱼ/∂zⱼ · ∂zⱼ/∂wᵢⱼ.\n\n"
            "Intermediate 'delta' values are reused across all weights in a layer, reducing complexity from O(W²) to O(W). "
            "Modern frameworks (PyTorch, TensorFlow) implement this via automatic differentiation, building a computation "
            "graph and applying the chain rule automatically.\n\n"
            "Common challenges include vanishing gradients in deep networks with sigmoid/tanh activations. Solutions include "
            "ReLU activations, batch normalization, residual connections (ResNets), and gradient clipping. Understanding "
            "backpropagation is essential for debugging training instabilities and designing novel architectures."
        ),
    },
    {
        "prompt": "What are activation functions and why is ReLU preferred over sigmoid in deep networks?",
        "response": (
            "Activation functions introduce non-linearity, allowing neural networks to learn complex mappings. Without them, "
            "stacking linear layers collapses to a single linear transformation regardless of depth. Common choices: "
            "sigmoid, tanh, ReLU (max(0,x)), Leaky ReLU, ELU, and GELU.\n\n"
            "Sigmoid causes vanishing gradients: its maximum gradient is 0.25, and it approaches zero for large |x|. In "
            "deep networks, gradients shrink exponentially during backpropagation, making early layers learn extremely "
            "slowly. ReLU has gradient 1 for positive inputs, avoiding this. It is also computationally cheap and induces "
            "sparsity (~half of neurons output zero), acting as implicit regularization.\n\n"
            "ReLU's weakness is 'dying ReLU': neurons receiving only negative inputs stop learning entirely. Leaky ReLU "
            "(gradient 0.01 for x<0) and ELU address this. GELU (used in BERT and GPT) applies a smooth probabilistic "
            "gate and often outperforms ReLU on NLP tasks."
        ),
    },
    # CNN
    {
        "prompt": "Why are convolutional neural networks so effective for image recognition?",
        "response": (
            "CNNs exploit three structural priors matched to images: local connectivity, weight sharing, and spatial "
            "hierarchy. Filters slide across the input computing local dot products, capturing patterns like edges and "
            "textures. Weight sharing — the same filter everywhere — reduces parameters dramatically and provides "
            "translation invariance.\n\n"
            "Pooling layers subsample feature maps, reducing spatial dimensions and increasing invariance. Stacking "
            "conv+pool blocks creates a hierarchy: early layers detect edges, middle layers detect parts, deep layers "
            "detect semantic concepts. This mirrors the hierarchical structure of visual content.\n\n"
            "Key innovations — AlexNet (deep CNNs + ReLU), ResNet (skip connections enabling 100+ layer networks), "
            "EfficientNet (compound scaling) — progressively improved accuracy. CNNs power object detection (YOLO), "
            "segmentation (U-Net), and medical imaging, demonstrating remarkable versatility."
        ),
    },
    # RNN/LSTM
    {
        "prompt": "What problem do LSTMs solve that vanilla RNNs cannot handle well?",
        "response": (
            "Vanilla RNNs suffer from the vanishing gradient problem during backpropagation through time (BPTT): gradients "
            "are multiplied by the recurrent weight matrix at each step, causing them to vanish or explode exponentially "
            "over long sequences. This makes it nearly impossible to learn dependencies spanning more than 10-20 steps.\n\n"
            "LSTMs introduce a cell state Cₜ — a memory highway with only additive and multiplicative interactions — "
            "allowing gradients to flow with minimal attenuation. Three gates control information flow: forget gate "
            "(what to erase), input gate (what to write), and output gate (what to expose as hidden state). All gates "
            "are learned end-to-end via backpropagation.\n\n"
            "GRUs simplify LSTMs by merging cell and hidden state with two gates (reset and update), achieving similar "
            "performance with fewer parameters. Both excel at language modeling, translation, and time-series forecasting. "
            "However, their sequential nature limits parallelism; Transformers with self-attention have largely superseded "
            "them for tasks requiring very long context."
        ),
    },
    # transformers
    {
        "prompt": "How does the self-attention mechanism in Transformers work?",
        "response": (
            "Self-attention computes a weighted sum of value vectors, where weights reflect how relevant each token is to "
            "every other token. For each token, three vectors are derived: Query (Q), Key (K), and Value (V). Attention "
            "scores: softmax(QKᵀ / √dₖ). Output: weighted sum of V. Scaling by √dₖ prevents vanishing gradients in high "
            "dimensions. Multi-head attention runs this in parallel with different projections.\n\n"
            "Unlike RNNs, self-attention has O(1) path length between any two tokens — a direct connection regardless of "
            "distance. This makes long-range dependencies easy to learn. The cost is O(n²) memory and compute (mitigated "
            "by sparse attention and FlashAttention). Transformers are fully parallelizable across positions, enabling "
            "efficient GPU training.\n\n"
            "The Transformer stacks encoder/decoder blocks: multi-head attention + feed-forward + layer norm + residual "
            "connections. Positional encodings inject order since attention is permutation-invariant. BERT (encoder-only) "
            "excels at understanding; GPT (decoder-only, causal attention) at generation; T5 (encoder-decoder) at seq2seq."
        ),
    },
    # reinforcement learning
    {
        "prompt": "What is the difference between model-free and model-based reinforcement learning?",
        "response": (
            "In RL, an agent learns to maximize cumulative reward through environment interaction. Model-free methods "
            "learn a policy or value function directly from experience without representing environment dynamics. "
            "Q-Learning and DQN learn Q(s,a) via TD updates; policy gradient methods (PPO, SAC) directly optimize the "
            "policy. They are simple but require many environment interactions.\n\n"
            "Model-based methods learn or use a world model P(s'|s,a) to plan ahead (e.g., Monte Carlo Tree Search in "
            "AlphaGo). Planning in imagination is far more sample-efficient since the agent generates synthetic "
            "experience without real rollouts. The risk is model bias: inaccurate dynamics lead to suboptimal policies.\n\n"
            "Modern hybrids — Dreamer, MuZero — learn compact latent-space world models and plan within them, achieving "
            "strong sample efficiency. Model-free is preferred for cheap simulators (game-playing); model-based is "
            "preferred when real-world interactions are expensive (robotics, drug discovery)."
        ),
    },
    # clustering
    {
        "prompt": "How do K-Means and DBSCAN differ in their approach to clustering?",
        "response": (
            "K-Means partitions data into K clusters by iteratively assigning points to the nearest centroid and "
            "recomputing centroids, minimizing within-cluster sum of squared distances. It requires K to be specified "
            "in advance and assumes convex, isotropic clusters of similar size. It is sensitive to initialization "
            "(K-Means++ helps) and outliers. Time complexity is O(nKd) per iteration, making it scalable.\n\n"
            "DBSCAN defines clusters as dense regions separated by low-density areas, using ε (neighborhood radius) "
            "and minPts (minimum points for a dense region). It discovers arbitrary-shaped clusters, automatically "
            "determines K, and labels outliers as noise — invaluable for anomaly detection. It struggles when clusters "
            "have varying density and requires careful parameter tuning.\n\n"
            "Use K-Means when K is known, clusters are spherical, and speed matters (customer segmentation, image "
            "compression). Use DBSCAN for arbitrary shapes, outlier detection, and unknown K (geospatial analysis, "
            "anomaly detection). HDBSCAN extends DBSCAN to handle varying densities and is often the best default "
            "for exploratory clustering."
        ),
    },
    # dimensionality reduction
    {
        "prompt": "What is PCA and how does it reduce dimensionality while preserving information?",
        "response": (
            "PCA projects high-dimensional data onto a lower-dimensional subspace defined by directions of maximum "
            "variance. It finds the eigenvectors (principal components) of the data covariance matrix. The first PC "
            "points in the direction of greatest variance, the second in the direction of greatest remaining variance "
            "orthogonal to the first, and so on. Data is projected onto the top-k eigenvectors.\n\n"
            "Information retained is measured by the explained variance ratio: sum of eigenvalues of selected PCs / "
            "total variance. Selecting k components explaining 95% of variance is a common heuristic. PCA is equivalent "
            "to SVD of the centered data matrix: X = UΣVᵀ, where columns of V are the principal components. Randomized "
            "SVD makes PCA scalable to very large datasets.\n\n"
            "PCA is used for visualization, noise reduction, feature extraction, and addressing the curse of "
            "dimensionality. Its limitation is linearity — it cannot capture non-linear manifold structure. t-SNE and "
            "UMAP excel at non-linear 2D visualization but are not suitable for production pipelines (non-deterministic, "
            "no out-of-sample extension for t-SNE)."
        ),
    },
    # model evaluation
    {
        "prompt": "Why is accuracy often a misleading metric for classification problems?",
        "response": (
            "Accuracy (correct / total) is intuitive but misleading for imbalanced datasets. A fraud detector that always "
            "predicts 'not fraud' on a dataset where 99% of transactions are legitimate achieves 99% accuracy while being "
            "completely useless. The metric hides failure on the minority class — which is often the one that matters.\n\n"
            "Precision = TP/(TP+FP) measures how many predicted positives are truly positive (minimizes false alarms). "
            "Recall = TP/(TP+FN) measures how many actual positives are found (minimizes missed detections). F1 is their "
            "harmonic mean, penalizing extreme imbalance. Fβ generalizes this: β>1 weights recall higher (medical "
            "diagnosis), β<1 weights precision higher.\n\n"
            "ROC-AUC summarizes performance across all thresholds and is robust to class imbalance. PR-AUC is more "
            "informative when the positive class is very rare. For multi-class problems, macro-averaging weights all "
            "classes equally; micro-averaging weights by frequency. Always choose metrics aligned with the business "
            "objective before modeling."
        ),
    },
    # overfitting
    {
        "prompt": "What causes overfitting in ML models and what are the main strategies to prevent it?",
        "response": (
            "Overfitting occurs when a model learns noise and spurious patterns instead of the true signal, resulting in "
            "poor generalization. It manifests as a large gap between training and validation performance. The root cause "
            "is excessive model complexity relative to training data size. The bias-variance tradeoff formalizes this: "
            "complex models have low bias but high variance.\n\n"
            "Regularization techniques constrain complexity: L1/L2 weight penalties discourage large weights; Dropout "
            "randomly zeros activations, forcing redundant representations; Batch Normalization normalizes activations "
            "and acts as a mild regularizer; Early stopping halts training when validation loss stops improving; Data "
            "augmentation artificially expands the training set with label-preserving transformations.\n\n"
            "Architecture choices also matter: reducing depth/width, using pooling, and weight tying all reduce capacity. "
            "Cross-validation provides reliable generalization estimates. For tabular data, tree ensembles with "
            "subsampling (Random Forests, XGBoost with colsample_bytree) naturally prevent overfitting through "
            "randomization and shallow base learners."
        ),
    },
    # transfer learning
    {
        "prompt": "How does transfer learning work and why is it effective for tasks with limited labeled data?",
        "response": (
            "Transfer learning leverages a model pre-trained on a large source task (e.g., ImageNet, GPT pre-training) "
            "and adapts it to a different but related target task. Features learned on large datasets — edges/textures "
            "for vision, syntactic/semantic patterns for NLP — are broadly useful. Starting from rich pre-trained "
            "representations instead of random initialization dramatically reduces the data needed.\n\n"
            "Fine-tuning has two stages: pre-train on a large dataset, then fine-tune on the target task. The backbone "
            "can be frozen (only training a new head — fast but less flexible), gradually unfrozen, or fully updated "
            "with a small learning rate. Freezing is preferred when target data is small and similar to the source; "
            "full fine-tuning is preferred when target data is large or very different.\n\n"
            "Transfer learning is transformative for NLP: BERT and GPT achieve state-of-the-art on downstream tasks "
            "with only hundreds of labeled examples. In computer vision, ImageNet pre-trained ResNets outperform "
            "random initialization on medical imaging with thousands of samples. The pre-trained model has learned a "
            "rich general-purpose feature space; fine-tuning adapts decision boundaries to the new task."
        ),
    },
    # LLM fine-tuning
    {
        "prompt": "What is instruction fine-tuning for LLMs and how does it differ from pre-training?",
        "response": (
            "LLMs are pre-trained on massive corpora with next-token prediction, learning rich language representations "
            "but not necessarily how to follow instructions helpfully. Instruction fine-tuning (IFT) adapts them by "
            "training on (instruction, response) pairs, teaching the model to produce helpful answers to diverse prompts.\n\n"
            "Training uses the same causal LM loss (cross-entropy) but applied only to response tokens — the instruction "
            "is masked. This teaches response generation without memorizing instructions. Data quality matters enormously: "
            "a few thousand high-quality pairs often outperform hundreds of thousands of noisy ones (LIMA: less is more).\n\n"
            "IFT is typically followed by RLHF or DPO to align with human preferences. For most practical applications "
            "— domain specialization, custom personas, format adherence — IFT alone is sufficient. Combined with QLoRA, "
            "it is feasible to fine-tune 7B+ parameter models on consumer hardware, as in this lab."
        ),
    },
    {
        "prompt": "What are the main challenges when fine-tuning large language models and how can they be mitigated?",
        "response": (
            "The most fundamental challenge is computational: a 7B parameter model in float32 requires ~28GB to store "
            "weights, and Adam optimizer states add another 2-4x. QLoRA addresses this by quantizing the base model to "
            "4-bit NF4 and training only small low-rank adapters, reducing memory by 10-20x while retaining most "
            "fine-tuning quality.\n\n"
            "Catastrophic forgetting is another challenge: fine-tuning on a narrow domain can degrade general capabilities. "
            "Mitigation strategies include small learning rates, fewer training steps, and mixing in general-purpose data. "
            "LoRA partially mitigates forgetting because only adapter weights change — original weights stay frozen.\n\n"
            "Data quality and format are critical. Inconsistent formatting, factual errors, or mismatched responses will "
            "be learned by the model. Careful curation and deduplication are essential. Gradient instability is common; "
            "paged_adamw_32bit (pages optimizer states to CPU RAM), cosine scheduling with warmup, and gradient clipping "
            "(max_grad_norm=0.3) are standard practices that stabilize training."
        ),
    },
    # QLoRA/LoRA
    {
        "prompt": "Can you explain LoRA and QLoRA and how they enable fine-tuning large models on limited hardware?",
        "response": (
            "LoRA (Low-Rank Adaptation) avoids updating all model weights by adding two small trainable matrices to each "
            "target layer: W₀ + ΔW = W₀ + BA, where B ∈ ℝᵐˣʳ and A ∈ ℝʳˣⁿ, rank r << min(m,n). Only A and B are "
            "trained; W₀ is frozen. For a 4096×4096 matrix with r=64, trainable params drop from 16.7M to 524K — 96.9% "
            "reduction. At inference, ΔW can be merged back: W = W₀ + BA, adding zero latency.\n\n"
            "QLoRA extends LoRA by quantizing the frozen base model to 4-bit NormalFloat (NF4), a data type designed "
            "for normally distributed weights. It adds double quantization (quantizing the quantization constants) and "
            "paged optimizers to handle memory spikes. This reduces VRAM for fine-tuning a 7B model from ~56GB (float16 "
            "full fine-tuning) to ~10-12GB, enabling training on a consumer GPU like the RTX 3090/4090.\n\n"
            "In this lab: r=64 (rank, dimensionality of adapter matrices), alpha=16 (scaling factor; effective scale = "
            "alpha/r = 0.25), dropout=0.1 (prevents adapter overfitting). Higher r captures more complex adaptations "
            "but uses more memory. LoRA is applied to attention projection matrices (q_proj, v_proj at minimum)."
        ),
    },
    # feature engineering
    {
        "prompt": "What is feature engineering and why is it still important in the age of deep learning?",
        "response": (
            "Feature engineering transforms raw data into representations that make ML algorithms work better. It includes "
            "creating new features (interaction terms, ratios, lag features), transforming existing ones (log-transforming "
            "skewed distributions, standardizing), encoding categoricals (one-hot, target encoding, embeddings), and "
            "selecting informative subsets (mutual information, recursive feature elimination).\n\n"
            "For traditional ML algorithms, feature engineering is often the single most impactful lever. A linear model "
            "with well-engineered features can outperform a neural network on structured data. Domain-specific features "
            "encode knowledge the algorithm cannot easily learn: a 'debt-to-income ratio' is far more informative for "
            "credit scoring than raw debt and income separately, even if the model could theoretically learn it.\n\n"
            "Deep learning reduces the need for manual feature engineering for unstructured data (images, text, audio). "
            "However, for tabular data — dominant in industry — deep learning does not consistently beat gradient "
            "boosting, and feature engineering remains critical. Even with deep learning, domain knowledge guides "
            "architecture: GNNs for molecular data, CNNs for spatial inputs, positional encodings for ordered sequences."
        ),
    },
    # MLOps
    {
        "prompt": "What is MLOps and why is it essential for deploying machine learning models in production?",
        "response": (
            "MLOps bridges ML development and production deployment, drawing from DevOps principles. A model with great "
            "offline metrics that cannot be reliably deployed, monitored, or updated is not production-ready. MLOps "
            "addresses the full lifecycle: data versioning, experiment tracking, model registry, CI/CD, serving "
            "infrastructure, and monitoring.\n\n"
            "ML systems have additional failure modes beyond traditional software: data drift (production input "
            "distributions shift from training), concept drift (input-output relationships change), training-serving skew "
            "(different preprocessing in training vs. inference), and feedback loops. Without monitoring, a model can "
            "silently degrade, causing business harm that is hard to diagnose.\n\n"
            "Key tools: MLflow or W&B for experiment tracking; DVC for data versioning; Airflow/Kubeflow for pipeline "
            "orchestration; feature stores (Feast) for consistent feature computation; Docker+Kubernetes for reproducible "
            "serving. The maturity model progresses from manual notebooks to automated CI/CD/CT (continuous training) "
            "pipelines that detect drift and trigger retraining automatically."
        ),
    },
    # ethics
    {
        "prompt": "What are the main sources of bias in AI systems and how can they be detected and mitigated?",
        "response": (
            "Bias arises at multiple pipeline stages. Historical bias: training data reflects past discrimination (e.g., "
            "hiring data underrepresenting women in technical roles). Representation bias: certain groups are "
            "underrepresented, causing poor performance for them. Measurement bias: features measured less accurately "
            "for some groups. Aggregation bias: a single model for a heterogeneous population ignores subgroup differences.\n\n"
            "Fairness metrics quantify different notions of fairness — which are mathematically incompatible in most "
            "settings: demographic parity (equal positive rates across groups), equalized odds (equal TPR and FPR), "
            "and individual fairness (similar individuals get similar predictions). The right metric is a value judgment "
            "that must involve domain experts and affected communities, not purely a technical decision.\n\n"
            "Mitigation strategies: pre-processing (resampling/reweighting to balance representation), in-processing "
            "(fairness constraints in the optimization objective, adversarial debiasing), and post-processing (adjusting "
            "per-group thresholds to equalize error rates). Model cards and datasheets document limitations and intended "
            "use. Ongoing monitoring for disparate impact in deployment is essential as population distributions shift."
        ),
    },
    # extra pairs to easily reach 60 samples
    {
        "prompt": "What is the curse of dimensionality and how does it affect machine learning?",
        "response": (
            "The curse of dimensionality refers to the exponential increase in data sparsity as the number of features "
            "grows. In high dimensions, all data points become approximately equidistant from each other, undermining "
            "distance-based algorithms like KNN and kernel methods. The volume of the feature space grows exponentially "
            "with dimensionality, so the same number of training samples covers an ever-smaller fraction of the space.\n\n"
            "Concretely: to maintain the same data density as you add dimensions, the number of required samples grows "
            "exponentially. A uniform grid with 10 points per dimension requires 10^d samples for d dimensions — 1000 "
            "for d=3, but 10^20 for d=20. This makes many models brittle in high dimensions: decision boundaries become "
            "unreliable, and overfitting is more likely.\n\n"
            "Mitigations include dimensionality reduction (PCA, autoencoders), feature selection (dropping irrelevant "
            "features), regularization (penalizing complex models), and using algorithms that handle high dimensions "
            "well (linear models, tree ensembles that select relevant features at each split). The bias-variance tradeoff "
            "becomes especially important: simpler models with higher bias may generalize better in high-dimensional "
            "settings than expressive models."
        ),
    },
    {
        "prompt": "What is cross-validation and why is it preferable to a single train/test split?",
        "response": (
            "Cross-validation (CV) is a resampling technique that provides a more reliable estimate of a model's "
            "generalization performance than a single train/test split. In k-fold CV, the dataset is split into k "
            "equal folds; the model is trained on k-1 folds and evaluated on the remaining fold, repeating k times. "
            "The final performance estimate is the average across all k evaluations.\n\n"
            "A single train/test split can give a misleading estimate because it depends heavily on which examples "
            "happen to land in the test set. If the test set is easy, performance is overestimated; if hard, it is "
            "underestimated. CV reduces this variance by testing on all data points exactly once. Stratified k-fold "
            "preserves class proportions in each fold, important for imbalanced datasets. Nested CV uses an outer "
            "loop for performance estimation and an inner loop for hyperparameter tuning, preventing data leakage.\n\n"
            "Leave-one-out CV (LOOCV) is the extreme case (k = n samples), providing low-bias estimates but high "
            "variance and high computational cost. For most practical purposes, k=5 or k=10 balances bias, variance, "
            "and computation. CV is especially important for small datasets where a fixed test set wastes valuable "
            "training data."
        ),
    },
    {
        "prompt": "What is batch normalization and how does it accelerate neural network training?",
        "response": (
            "Batch Normalization (BN, Ioffe & Szegedy 2015) normalizes the activations of each layer to have zero mean "
            "and unit variance within each mini-batch, then applies learnable scale (γ) and shift (β) parameters: "
            "BN(x) = γ · (x - μ_B) / σ_B + β. This is applied before or after the non-linearity, depending on the "
            "architecture convention.\n\n"
            "BN accelerates training in several ways: it reduces internal covariate shift (the change in input "
            "distribution of each layer as earlier layers' parameters update), allowing higher learning rates without "
            "divergence. It also has a mild regularization effect — the noise from mini-batch statistics acts like "
            "dropout. Models with BN are less sensitive to initialization, making training more robust.\n\n"
            "At inference, batch statistics are replaced by running averages computed during training (population "
            "mean and variance), making predictions deterministic. BN is less effective with very small batch sizes "
            "(batch size 1-2) because the batch statistics are noisy. Alternatives include Layer Normalization "
            "(normalizes across features for a single sample — preferred in Transformers) and Group Normalization "
            "(normalizes groups of channels — preferred in object detection with small batches)."
        ),
    },
    {
        "prompt": "What is the attention mechanism and how is it used in sequence-to-sequence models?",
        "response": (
            "The attention mechanism (Bahdanau et al. 2015) was introduced to address the bottleneck in vanilla "
            "encoder-decoder architectures, where the entire input sequence is compressed into a single fixed-size "
            "context vector. Attention allows the decoder to selectively focus on different parts of the encoder's "
            "output at each decoding step, computing a weighted sum of encoder hidden states.\n\n"
            "At each decoder step t, attention scores are computed between the decoder's current hidden state and all "
            "encoder states: eₜᵢ = score(sₜ, hᵢ). Scores are normalized with softmax to produce attention weights αₜᵢ. "
            "The context vector cₜ = ∑ αₜᵢ hᵢ is concatenated with the decoder input. Score functions include dot "
            "product (fast), additive (Bahdanau), and multiplicative (Luong) variants.\n\n"
            "Attention enables models to handle long sequences where vanilla RNNs fail, and the attention weights "
            "provide interpretable alignments (e.g., which source words correspond to each target word in translation). "
            "The Transformer architecture scales this idea by replacing recurrence entirely with multi-head self-attention, "
            "enabling full parallelism and achieving state-of-the-art on virtually all sequence modeling tasks."
        ),
    },
    {
        "prompt": "What is the vanishing gradient problem and how do residual connections address it?",
        "response": (
            "The vanishing gradient problem occurs during backpropagation in deep networks: gradients are multiplied "
            "by weight matrices and activation function derivatives at each layer. When these multipliers are "
            "consistently less than 1 (as with sigmoid/tanh activations), gradients shrink exponentially with depth, "
            "making early layers receive near-zero gradient signals and learn extremely slowly or not at all.\n\n"
            "Residual connections (He et al. 2016, ResNet) address this with a simple architectural change: instead "
            "of learning a mapping F(x), the layer learns a residual F(x) + x (the skip connection adds the input "
            "directly to the output). During backpropagation, the gradient flows through two paths: through F(x) "
            "(which may vanish) and directly through the skip connection (gradient = 1, no decay). This identity "
            "shortcut ensures that gradient signal is preserved regardless of depth.\n\n"
            "Residual connections enabled training of networks 100+ layers deep, with ResNet-152 winning ILSVRC 2015. "
            "They also initialize well (F(x) ≈ 0 initially, so the network starts as identity mappings) and provide "
            "implicit regularization. The concept generalizes to Highway Networks, DenseNet (skip connections from "
            "every layer to every subsequent layer), and Transformer blocks (which use residual connections around "
            "both the attention and feed-forward sublayers)."
        ),
    },
    {
        "prompt": "What is data augmentation and how does it improve model generalization?",
        "response": (
            "Data augmentation artificially expands the training set by applying label-preserving transformations to "
            "existing examples. For images: random flips, rotations, crops, color jitter, Gaussian noise, CutMix "
            "(replacing patches with patches from another image), and MixUp (linearly interpolating two images and "
            "their labels). For text: synonym replacement, random deletion, back-translation. For audio: pitch shift, "
            "time stretching, adding background noise.\n\n"
            "Augmentation improves generalization by exposing the model to a wider variety of training examples, "
            "reducing the risk of learning spurious patterns specific to the exact training images (e.g., a classifier "
            "that relies on the cat always appearing in the center). It effectively increases the training set size "
            "without collecting new labeled data — critical in domains where labeling is expensive (medical imaging, "
            "satellite imagery).\n\n"
            "The choice of augmentation must be domain-appropriate: horizontally flipping chest X-rays is fine (lungs "
            "are symmetric), but vertically flipping them changes anatomy meaningfully. Too aggressive augmentation "
            "can degrade performance by creating unrealistic examples the model cannot learn from. Modern AutoAugment "
            "and RandAugment policies search for the best augmentation strategy for a given dataset automatically."
        ),
    },
    {
        "prompt": "What is hyperparameter tuning and what are the main strategies for it?",
        "response": (
            "Hyperparameters are configuration values set before training that control the learning process: learning "
            "rate, batch size, number of layers, regularization strength, etc. Unlike model parameters (weights), "
            "they are not learned from data. Hyperparameter tuning is the process of searching for the combination "
            "that maximizes validation performance.\n\n"
            "Grid search exhaustively evaluates all combinations in a predefined grid — thorough but exponentially "
            "expensive. Random search samples combinations randomly; surprisingly, it often outperforms grid search "
            "with the same budget because it explores more distinct values per hyperparameter (Bergstra & Bengio, 2012). "
            "Bayesian optimization models the performance surface with a probabilistic surrogate (e.g., Gaussian Process) "
            "and uses an acquisition function (e.g., Expected Improvement) to select the next configuration to evaluate, "
            "concentrating evaluations in promising regions.\n\n"
            "Modern approaches include Hyperband (early stopping of poorly performing configurations), ASHA (Asynchronous "
            "Successive Halving), and Population-Based Training (evolving hyperparameters during training). Tools like "
            "Optuna, Ray Tune, and W&B Sweeps automate the process. Learning rate is typically the most impactful "
            "hyperparameter; a grid search over just the learning rate (log scale: 1e-5 to 1e-1) often captures most "
            "of the available performance gain."
        ),
    },
    {
        "prompt": "What is the difference between generative and discriminative models in machine learning?",
        "response": (
            "Discriminative models learn the conditional distribution P(y|x) — the probability of label y given input x. "
            "They directly model the decision boundary between classes. Examples: logistic regression, SVMs, neural "
            "network classifiers, CRFs. They are typically more accurate for classification when sufficient labeled "
            "data is available because they focus capacity on the decision boundary rather than modeling the full "
            "input distribution.\n\n"
            "Generative models learn the joint distribution P(x,y) or the class-conditional density P(x|y), from which "
            "P(y|x) can be derived via Bayes' theorem. Examples: Naive Bayes, Gaussian Mixture Models, VAEs, GANs, "
            "diffusion models. They can generate new samples from the learned distribution and handle missing data "
            "naturally. They require modeling the full input space, which is harder in high dimensions.\n\n"
            "The tradeoff is data efficiency vs. accuracy: generative models can use unlabeled data and work well "
            "with limited labels (semi-supervised learning). Discriminative models are more accurate with abundant "
            "labeled data. Modern LLMs like GPT are generative (model P(x)) and adapt to classification by framing "
            "it as text generation, blurring the traditional boundary. Foundation models pre-trained generatively "
            "then fine-tuned discriminatively combine the strengths of both paradigms."
        ),
    },
    {
        "prompt": "What is knowledge distillation and how is it used to compress large models?",
        "response": (
            "Knowledge distillation (Hinton et al. 2015) compresses a large, accurate 'teacher' model into a smaller, "
            "faster 'student' model by training the student to mimic the teacher's output distribution rather than just "
            "matching hard labels. The student is trained on a combination of cross-entropy with true labels and "
            "KL-divergence with the teacher's soft predictions (class probabilities). Soft predictions carry 'dark "
            "knowledge': the teacher's uncertainty and inter-class similarities (e.g., a 7 looks a bit like a 1).\n\n"
            "Temperature scaling softens the teacher's distribution: P_T(c) = exp(z_c/T) / ∑exp(z_j/T), where T>1 "
            "produces softer probabilities that provide richer training signal. The student typically achieves much "
            "better accuracy than training on hard labels alone, despite being smaller than the teacher.\n\n"
            "Distillation is widely used in production: DistilBERT achieves 97% of BERT's performance with 40% fewer "
            "parameters and 60% faster inference. Variants include feature distillation (matching intermediate "
            "representations), attention distillation (matching attention maps), and self-distillation (using the "
            "model's own predictions as targets). It is especially valuable when inference latency and compute cost "
            "are constrained, as in mobile and edge deployments."
        ),
    },
    {
        "prompt": "What is the role of the learning rate in training neural networks, and how should it be scheduled?",
        "response": (
            "The learning rate (η) is the most critical hyperparameter in neural network training. It controls the "
            "step size in gradient descent: w ← w - η · ∇L(w). Too high: training diverges or oscillates. Too low: "
            "training converges extremely slowly, potentially getting stuck in poor local minima or saddle points. "
            "The optimal learning rate depends on the optimizer, batch size, model architecture, and loss landscape.\n\n"
            "Learning rate scheduling adjusts η during training to improve convergence. Warmup linearly increases η "
            "from 0 to the target value over the first few percent of training steps — important for Transformers "
            "where large initial gradients can destabilize attention layers. Cosine annealing decays η following a "
            "cosine curve from the initial value to near zero, providing a smooth decay that avoids abrupt changes. "
            "Step decay reduces η by a factor (e.g., 10x) at fixed milestones. Cyclic LR oscillates between bounds, "
            "helping escape local minima.\n\n"
            "The learning rate range test (Smith, 2017) trains for a few iterations with η increasing exponentially "
            "and plots loss vs. η — the optimal range is where loss decreases most steeply. AdaGrad, RMSprop, and "
            "Adam adapt per-parameter learning rates based on gradient history, making them less sensitive to the "
            "global learning rate choice. In this lab, cosine scheduling with 3% warmup (warmup_ratio=0.03) is "
            "specified — a well-validated recipe for LLM fine-tuning."
        ),
    },
    {
        "prompt": "What is semi-supervised learning and when is it beneficial?",
        "response": (
            "Semi-supervised learning (SSL) leverages both labeled and unlabeled data to improve model performance. "
            "Labeled data is typically expensive and scarce (requires human annotation), while unlabeled data is "
            "abundant. SSL exploits the structure of unlabeled data — clusters, manifolds, density — to constrain "
            "the learned function and improve generalization beyond what supervised learning on the labeled set alone "
            "can achieve.\n\n"
            "Core SSL approaches include self-training (train on labeled data, generate pseudo-labels for unlabeled "
            "data with high-confidence predictions, retrain on the combined set — repeat), consistency regularization "
            "(predict similar outputs for augmented versions of the same unlabeled input — MixMatch, FixMatch), "
            "and graph-based methods (propagate labels through a similarity graph). Contrastive learning (SimCLR, "
            "MoCo) pre-trains representations by pulling augmented views of the same image together in embedding "
            "space, then fine-tunes on labeled data.\n\n"
            "SSL is most beneficial when labeled data is very scarce relative to the problem complexity — medical "
            "imaging (labeling requires expert radiologists), speech recognition (transcription is expensive), and "
            "NLP (domain-specific annotations). It is less beneficial when labeled data is abundant, as the "
            "additional complexity of SSL may not justify the gain. Foundation models pre-trained on massive unlabeled "
            "corpora (GPT, CLIP) represent the ultimate expression of SSL at scale."
        ),
    },
    {
        "prompt": "What is an autoencoder and what are its main applications?",
        "response": (
            "An autoencoder is a neural network trained to reconstruct its input through a bottleneck. It has two "
            "components: an encoder f that maps input x to a latent representation z = f(x) (typically lower-dimensional), "
            "and a decoder g that reconstructs the input from z: x̂ = g(z). Training minimizes reconstruction loss "
            "(e.g., MSE or binary cross-entropy between x and x̂). The bottleneck forces the encoder to learn a compact, "
            "informative representation.\n\n"
            "Applications are diverse. Dimensionality reduction: autoencoders learn non-linear compressions that PCA "
            "cannot. Anomaly detection: a model trained on normal data has high reconstruction error for anomalous "
            "inputs — useful for fraud detection and industrial defect detection. Denoising: denoising autoencoders "
            "(DAE) are trained to reconstruct clean inputs from corrupted ones, learning robust features. Generative "
            "modeling: Variational Autoencoders (VAEs) impose a probabilistic prior on the latent space (z ~ N(0,I)), "
            "enabling sampling of new data points.\n\n"
            "Sparse autoencoders, used in mechanistic interpretability of LLMs, decompose activations into a large "
            "dictionary of sparse, interpretable features — revealing that neural networks represent concepts as "
            "superpositions of directions in activation space. The basic autoencoder framework has been foundational "
            "to much of modern deep learning, from pre-training strategies to diffusion model architectures."
        ),
    },
    {
        "prompt": "What is the difference between online learning and batch learning in machine learning?",
        "response": (
            "Batch learning trains a model on the entire training dataset at once, then deploys it as a static model. "
            "It is the standard approach for offline tasks where all training data is available upfront and the data "
            "distribution is assumed to be stationary. It typically achieves better final performance because the model "
            "sees the full data distribution during training. However, retraining from scratch when new data arrives "
            "is computationally expensive and slow.\n\n"
            "Online learning updates the model incrementally as each new example (or mini-batch) arrives, without "
            "retaining the full training history. It is essential for streaming data, non-stationary distributions "
            "(where the relationship between inputs and outputs changes over time), and memory-constrained environments. "
            "Algorithms like Stochastic Gradient Descent (SGD), online logistic regression, and Vowpal Wabbit are "
            "designed for this setting. The challenge is catastrophic forgetting: learning from new data tends to "
            "overwrite knowledge from old data.\n\n"
            "Continual learning (also called lifelong learning) is the subfield that studies how to update models "
            "incrementally while retaining prior knowledge. Approaches include Elastic Weight Consolidation (EWC, which "
            "penalizes changes to weights important for previous tasks), progressive neural networks (adding new columns "
            "for new tasks), and replay buffers (storing a small subset of old data to interleave with new data). "
            "In production, periodic batch retraining triggered by concept drift detection is a common pragmatic solution."
        ),
    },
    {
        "prompt": "How do support vector machines (SVMs) work and what is the kernel trick?",
        "response": (
            "SVMs find the maximum-margin hyperplane that separates two classes. The margin is the distance between the "
            "hyperplane and the closest training points from each class (support vectors). Maximizing the margin "
            "minimizes VC dimension, providing theoretical guarantees on generalization. The optimization problem is "
            "convex (quadratic programming), guaranteeing a global optimum. The soft-margin SVM introduces slack "
            "variables to allow misclassifications, controlled by hyperparameter C: high C = low tolerance for "
            "misclassifications (complex boundary), low C = large margin with more misclassifications.\n\n"
            "The kernel trick enables SVMs to learn non-linear decision boundaries without explicitly computing "
            "high-dimensional feature maps. A kernel function K(x,y) computes the dot product of two inputs in a "
            "high-dimensional (possibly infinite-dimensional) feature space implicitly: K(x,y) = φ(x)·φ(y). Common "
            "kernels: RBF (Gaussian, K(x,y) = exp(-γ||x-y||²)), polynomial (K(x,y) = (x·y + c)^d), and sigmoid. "
            "The choice of kernel encodes prior knowledge about the data's structure.\n\n"
            "SVMs were state-of-the-art for many classification tasks before deep learning. They remain competitive "
            "for small-to-medium datasets, especially with the RBF kernel, and are theoretically well-understood. "
            "Their main limitations are poor scalability to very large datasets (O(n²-n³) training time), sensitivity "
            "to feature scaling, and difficulty extending to multi-class problems (requires one-vs-one or one-vs-all "
            "schemes). For large tabular datasets, gradient boosting is usually preferred."
        ),
    },
    {
        "prompt": "What is the difference between precision and recall, and how do you decide which to optimize?",
        "response": (
            "Precision measures the fraction of positive predictions that are actually positive: TP/(TP+FP). It answers: "
            "when the model says 'positive', how often is it right? Recall (sensitivity) measures the fraction of actual "
            "positives that are correctly identified: TP/(TP+FN). It answers: of all actual positives, how many did the "
            "model find? There is an inherent tradeoff: lowering the classification threshold increases recall (more "
            "positives caught) but decreases precision (more false alarms).\n\n"
            "The decision of which to optimize depends entirely on the cost asymmetry between false positives and false "
            "negatives in the specific application. For cancer screening, missing a positive case (false negative) is "
            "catastrophic — the patient goes untreated. High recall is paramount, even at the cost of many false "
            "positives (biopsies of benign tissue). For spam filtering, incorrectly marking a legitimate email as spam "
            "(false positive) damages user trust more than missing a spam email. High precision is preferred.\n\n"
            "F1-score (harmonic mean of precision and recall) provides a single metric when both matter equally. Fβ "
            "score generalizes this: β>1 weights recall more heavily (β=2: recall counts twice as much as precision), "
            "β<1 weights precision. The PR curve plots precision vs. recall at all thresholds; its AUC summarizes "
            "overall model quality. Always align your optimization target with business requirements before modeling — "
            "this choice should drive both metric selection and threshold tuning at deployment time."
        ),
    },
    {
        "prompt": "What is ensemble learning and what are its main strategies?",
        "response": (
            "Ensemble learning combines multiple models (base learners) to produce a stronger predictor than any "
            "individual model. The central insight is that diverse models make uncorrelated errors; averaging their "
            "predictions reduces variance without increasing bias (for averaging) or reduces bias while controlling "
            "variance (for boosting). Ensembles routinely win machine learning competitions and power production systems.\n\n"
            "The three main strategies are: Bagging (Bootstrap Aggregating) trains each base learner on a random "
            "bootstrap sample of the training data and aggregates predictions by voting or averaging — Random Forests "
            "are the canonical example, adding feature randomness to further decorrelate trees. Boosting trains "
            "learners sequentially, each correcting the errors of the previous ensemble — AdaBoost reweights "
            "misclassified examples, while Gradient Boosting fits residuals. Stacking trains a meta-learner to "
            "combine base learner predictions, using out-of-fold predictions to avoid leakage.\n\n"
            "Diversity is essential: homogeneous ensembles of identical models gain little. Diversity is achieved "
            "via data sampling (bagging), feature sampling (Random Forests), different algorithms (stacking linear "
            "regression + random forest + XGBoost), different hyperparameters, or different random seeds. The bias-"
            "variance decomposition explains why ensembles work: independent models with error ε have ensemble error "
            "ε/n (variance term), while correlated models provide less improvement. Test-time augmentation (averaging "
            "predictions over augmented versions of the test input) applies ensembling principles to single models."
        ),
    },
    {
        "prompt": "What is object detection and how do modern detectors like YOLO work?",
        "response": (
            "Object detection simultaneously classifies objects in an image and localizes them with bounding boxes. "
            "Unlike image classification (one label per image), detection must handle multiple objects at different "
            "scales and positions. Early detectors (R-CNN family) used a two-stage approach: first generate region "
            "proposals (candidate bounding boxes), then classify each region. While accurate, this is slow.\n\n"
            "YOLO (You Only Look Once) reformulates detection as a single regression problem: the image is divided "
            "into an S×S grid; each cell predicts B bounding boxes (coordinates + confidence) and C class "
            "probabilities. Everything is predicted in one forward pass, making YOLO extremely fast (real-time "
            "at 45+ FPS). Anchor boxes — predefined aspect ratios — help the network predict boxes of different "
            "shapes. Non-Maximum Suppression (NMS) removes duplicate detections by keeping the highest-confidence "
            "box when overlapping boxes (IoU > threshold) predict the same object.\n\n"
            "Modern YOLO versions (v5-v10, YOLO-NAS) and rivals like EfficientDet add feature pyramid networks "
            "(FPN) for multi-scale detection, deformable convolutions, and attention mechanisms. Evaluation uses "
            "mean Average Precision (mAP) at different IoU thresholds (mAP@0.5, COCO mAP@[0.5:0.95]). Object "
            "detection powers self-driving cars, surveillance, medical imaging, and robotics."
        ),
    },
    {
        "prompt": "What is natural language processing (NLP) and what are its core tasks?",
        "response": (
            "Natural Language Processing (NLP) is the subfield of AI concerned with enabling computers to understand, "
            "process, and generate human language. Language is complex: it is ambiguous (bank means river bank or "
            "financial institution depending on context), compositional (meaning emerges from the structure of words), "
            "and pragmatic (what is said differs from what is meant). NLP bridges the gap between unstructured text "
            "and machine-interpretable representations.\n\n"
            "Core NLP tasks include: text classification (sentiment analysis, spam detection, topic categorization), "
            "named entity recognition (identifying persons, organizations, locations in text), relation extraction "
            "(finding relationships between entities), machine translation (translating between languages), "
            "question answering (extracting or generating answers from a context), summarization (abstractive or "
            "extractive condensation of text), and language modeling (predicting the next token in a sequence, "
            "the foundation of LLMs like GPT).\n\n"
            "The dominant paradigm shifted from rule-based systems (1960s-1990s) to statistical methods (1990s-2010s, "
            "n-gram models, SVMs on TF-IDF features) to neural methods (2010s, word2vec, LSTMs) to Transformer-based "
            "pre-trained models (2018-present: BERT, GPT, T5, LLaMA). Modern LLMs achieve remarkable performance "
            "across all NLP tasks through in-context learning and instruction following, largely without task-specific "
            "architectures."
        ),
    },
    {
        "prompt": "What are word embeddings and how do Word2Vec and GloVe learn them?",
        "response": (
            "Word embeddings are dense, low-dimensional vector representations of words that capture semantic and "
            "syntactic relationships. Unlike one-hot encoding (sparse, no notion of similarity), embeddings place "
            "semantically related words close together in vector space. The famous example: vector('king') - "
            "vector('man') + vector('woman') ≈ vector('queen'), demonstrating that arithmetic on embeddings "
            "reflects semantic relationships.\n\n"
            "Word2Vec (Mikolov et al. 2013) trains shallow neural networks on large corpora using two tasks: "
            "CBOW (Continuous Bag of Words — predict a word from its context) or Skip-gram (predict context words "
            "given a center word). The hidden layer weights become the word embeddings. Negative sampling makes "
            "training efficient by only updating weights for the true word and a small number of random 'negative' "
            "words, avoiding the expensive softmax over the full vocabulary.\n\n"
            "GloVe (Global Vectors, Pennington et al. 2014) takes a different approach: it constructs a global "
            "word-word co-occurrence matrix and factorizes it, optimizing embeddings to predict co-occurrence "
            "probabilities. It captures global corpus statistics that Word2Vec's local window approach misses. "
            "Both produce similar quality embeddings in practice. Modern contextual embeddings (ELMo, BERT) produce "
            "different vectors for the same word in different contexts — 'bank' has different embeddings in 'river "
            "bank' and 'bank account' — overcoming the fundamental limitation of static embeddings."
        ),
    },
    {
        "prompt": "What is the difference between supervised and unsupervised learning?",
        "response": (
            "Supervised learning trains a model using labeled examples — pairs (x, y) where x is the input and y "
            "is the target label or value provided by a human annotator. The model learns a mapping f: x → y by "
            "minimizing a loss function that measures the discrepancy between predictions and labels. Classification "
            "(predicting discrete classes) and regression (predicting continuous values) are the two main supervised "
            "tasks. Supervised learning requires labeled data, which can be expensive to collect.\n\n"
            "Unsupervised learning finds patterns in data without labels. The model discovers structure inherent "
            "in the input distribution: clusters (K-Means, DBSCAN, GMMs), latent representations (autoencoders, "
            "PCA, ICA), density estimates (kernel density estimation, normalizing flows), or generative models "
            "(VAEs, GANs). Unlabeled data is abundant and cheap, making unsupervised learning attractive for "
            "large-scale representation learning.\n\n"
            "The boundary has blurred with self-supervised learning, where labels are generated automatically from "
            "the data itself (e.g., predicting masked tokens in BERT, predicting the next token in GPT, predicting "
            "the original image from an augmented version in contrastive learning). Self-supervised pre-training "
            "on massive unlabeled corpora, followed by supervised fine-tuning on small labeled datasets, is the "
            "dominant paradigm in modern NLP and computer vision. Semi-supervised learning further combines a small "
            "labeled set with a large unlabeled set."
        ),
    },
    {
        "prompt": "What is Bayesian inference and how does it differ from frequentist approaches in machine learning?",
        "response": (
            "Bayesian inference treats model parameters θ as random variables with a prior distribution P(θ) "
            "reflecting beliefs before seeing data. Given data D, Bayes' theorem updates this to the posterior: "
            "P(θ|D) = P(D|θ)·P(θ) / P(D). Predictions are made by integrating over the posterior: "
            "P(y|x,D) = ∫ P(y|x,θ) P(θ|D) dθ. This naturally quantifies uncertainty — the posterior distribution "
            "captures not just the best parameters but the full range of plausible parameter values.\n\n"
            "Frequentist approaches treat parameters as fixed unknowns estimated from data (MLE: find θ that "
            "maximizes P(D|θ); MAP: maximize the posterior mode). They provide point estimates without uncertainty "
            "quantification. Regularization (L2 penalty) corresponds to a Gaussian prior in the MAP framework, "
            "connecting regularized MLE to Bayesian inference. Frequentist methods are computationally simpler "
            "and dominate practice; Bayesian methods are more principled but require intractable integrals.\n\n"
            "Approximate inference methods make Bayesian ML practical: MCMC (Markov Chain Monte Carlo) samples "
            "from the posterior; Variational Inference approximates the posterior with a tractable distribution "
            "by minimizing KL divergence; Laplace approximation fits a Gaussian around the MAP estimate. Bayesian "
            "Neural Networks (BNNs) provide uncertainty estimates valuable in safety-critical applications "
            "(medical diagnosis, autonomous driving). Gaussian Processes are fully Bayesian non-parametric models "
            "widely used for regression and hyperparameter optimization (Bayesian optimization)."
        ),
    },
]


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Repeat pairs if needed to reach N_SAMPLES=60, then shuffle
    pool = PAIRS.copy()
    while len(pool) < 60:
        pool.extend(PAIRS)
    rng.shuffle(pool)
    pairs = pool[:60]

    if len(pairs) < 50:
        raise RuntimeError(f"Only {len(pairs)} pairs. Minimum required is 50.")

    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

    for path, subset in [(train_path, train_pairs), (test_path, test_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"Dataset saved to '{OUTPUT_DIR}/':\n"
        f"  train : {len(train_pairs)} pairs → {train_path}\n"
        f"  test  : {len(test_pairs)} pairs → {test_path}\n"
    )


if __name__ == "__main__":
    main()
