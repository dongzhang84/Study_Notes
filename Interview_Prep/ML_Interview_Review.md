# ML Interview Review (Fundamentals → Deep Learning)

> **How to use this**: A study sheet for US interviews. Part 1 walks through your resume/experience (interviewers dig here). Part 2 is a systematic ML knowledge review, basics → deep learning. Goal for each point: **explain the essence in 1–2 sentences + know when to use it** — not memorize formulas.
>
> Part 1 is filled from your resume — real projects, real numbers, turned into talking points + likely follow-ups. Correct or add color where you remember more detail; the metrics here are the resume's, so keep them consistent with what you say out loud.

---

# Part 1 — Resume / Experience Walkthrough

> Principle: **every resume line must survive 3 layers of follow-up** — "What did you do?" → "Why did you choose that?" → "What would you do differently / how did you evaluate it?"

## 1️⃣ Amazon — Machine Learning Scientist (Feb 2022 – Jul 2025)

**One-line:** Owned fraud-detection ML end-to-end at Amazon scale — from LLM fine-tuning and multi-task deep models to the MLOps platform behind them.

**Projects to tell** (each is a STAR story — situation → action → result):
- **LLM for fraud classification (LoRA).** Text-heavy fraud signals weren't well served by gradient-boosted trees. Fine-tuned LLMs with LoRA → **+15% F1 over GBDT**; shipped with **quantization** for production inference. *Why LoRA:* cheap to train/serve vs full fine-tune. *How evaluated:* F1 on held-out fraud, within a latency budget.
- **20+ production fraud models.** Tree-based + deep nets (PyTorch), **millions of transactions/day** across global retail and Just Walk Out stores, **sub-100ms** inference. You owned the full lifecycle at scale.
- **Multi-task learning model.** Consolidated several fraud tasks into one network — **shared embeddings + task-specific heads** → better generalization and **−60% training cost**. *Why MTL:* related tasks share signal, one model to maintain. *Risk you watched:* negative transfer, per-task metrics.
- **Global unified model on SageMaker.** One unified model replaced per-region models → **Try Before You Buy non-payment fraud −50%** across US/EU/Japan.
- **MLOps platform.** Data-drift detection, performance-degradation alerts, CI/CD retraining, feature store → **−70% maintenance overhead**.

**Frame it as:** large-scale, high-bar, production owner, comfortable from modeling to monitoring. ⚠️ Only talk about what you gained; never voice negativity about Amazon.

**Likely questions + how to answer:**
- *"Walk me through a model you shipped end to end."* → pick the unified model or MTL; go data → features → train → offline eval → deploy → monitor → iterate.
- *"Why LoRA over full fine-tuning, and why beat GBDT?"* → memory/cost of LoRA; LLMs read text signals trees miss; quantify +15% F1.
- *"Fraud drift / what broke in prod?"* → labels are delayed (chargebacks) + adversarial drift; drift detection + retraining triggers + rollback.
- *"How did you set the decision threshold?"* → business cost curve + calibrated scores → target precision/recall, not 0.5.

## 2️⃣ IBM — Data Scientist (Mar 2020 – Feb 2022)

**One-line:** Enterprise data science across support operations — regression, NLP text classification, an early-warning classifier, and a recommender — shipped with MLOps and validated with real experiments.

**Projects to tell:**
- **Work-order duration regression.** Predicted onsite repair duration for all of IBM's technical-support onsite services; built MLOps to auto-update and retrain the model. *Be ready on:* feature choices, why regression, how validated.
- **NPS Early Warning System.** Binary classifier on the support ticketing system → **+70% recall** over the previous model, validated with **A/B testing + hypothesis testing** (not just offline metrics). Your strongest "measured real impact" story — recall matters because the cost of missing an at-risk account is high.
- **Text classification (Kenexa BrassRing email).** Linear SVC baseline → Watson NLC / ANN models. Shows baseline-first discipline and the classic NLP progression.
- **Recommendation system for IBM business partners.** Rebuilt the data pipeline (**Spark, SQL**), added **time-series forecasting features** → **+30% recommendation relevance**.
- **Mentored 6+ junior data scientists.** Seniority / leadership signal.

**Frame it as:** enterprise ML with rigor — baseline-first, experiment-validated, MLOps-shipped.

**Likely questions + how to answer:**
- *"How did you know the NPS model actually helped?"* → A/B + hypothesis test, +70% recall, what recall means here.
- *"Why a Linear SVC baseline?"* → fast, interpretable, sets the bar before heavier models.
- *"How did you evaluate the recommender?"* → relevance lift (+30%), offline ranking + business signal; time-series features captured trend/seasonality.

## 3️⃣ AceRocket — Founder & CEO (2025 – Present)

**One-line:** Built and shipped an AI adaptive-learning SaaS for SAT/ACT/AP math 0→1 — full-stack product + LLM content pipeline + adaptive recommendation. This is your modern GenAI + product-ownership story, the best fit for LLM/agentic roles.

**What the AI/ML actually does:**
- **Adaptive recommendation.** Real-time performance analytics with topic/subtopic **mastery tracking** → recommends the next problem and surfaces knowledge gaps. Framed as ML: an **adaptive-learning / recommendation** problem — item difficulty + student ability (IRT / knowledge tracing), cold start for new students.
- **LLM content pipeline.** A **1500+ problem** question bank with an automated **LLM-powered pipeline (RAG + prompt orchestration)** for content, plus behavioral analytics to optimize learning paths.
- **Full-stack production SaaS.** Next.js/TypeScript frontend + Firebase backend (Auth, Firestore, Realtime DB); deployed and iterated on user analytics + tutoring feedback.

**Frame it as:** 0→1 shipping, cost/quality control, and owning the whole stack.

**Likely questions + how to answer:**
- *"How do you know a generated question is good?"* → no single ground truth → tutor/human eval + LLM-as-judge + difficulty calibrated against real student performance.
- *"How do you personalize the next problem?"* → per topic/subtopic mastery → adaptive recommendation; cold start → start from a diagnostic / popularity prior.
- *"How do you control hallucination and cost?"* → RAG grounding on a vetted bank, prompt orchestration, validation checks on generated items, caching.

## 4️⃣ Research background (physics / astrophysics PhD) — keep it brief

Astrophysics PhD (Ohio State, 2015), 17 peer-reviewed publications (15 first-author), 1300+ citations. Large-scale **numerical simulations** (radiation hydrodynamics, HPC/parallel computing), Monte Carlo, statistical modeling. **One-line pitch**: *"A physics PhD trained me to model messy problems from first principles, quantify uncertainty, and run large-scale numerical computation — the same instincts I bring to ML."* Don't over-explain the astrophysics; pivot quickly to the transferable skills (PDEs/numerical methods ↔ optimization & gradient descent; statistical mechanics ↔ softmax/max-entropy; Monte Carlo ↔ MCMC/Bayesian inference).

---

# Part 2 — Systematic ML Review (Fundamentals → Deep Learning)

## A. ML Foundations

**Supervised / Unsupervised / Reinforcement**
- Supervised: labeled data, learn X→y (classification, regression).
- Unsupervised: no labels, find structure (clustering, dimensionality reduction, density estimation).
- Reinforcement: agent interacts with environment to maximize cumulative reward.
- Semi-/self-supervised: few labels + lots of unlabeled / build supervision from the data itself (the core of modern LLM pretraining).

**Bias–Variance Tradeoff (must-know)**
- Bias: model too simple → underfitting → bad on both train and test.
- Variance: model too complex → overfitting → good on train, bad on test.
- Total error = bias² + variance + irreducible noise. Goal: find the balance.
- One-liner: **underfit → add complexity/features; overfit → add regularization/data or reduce complexity.**

**Handling overfitting**: more data, data augmentation, regularization (L1/L2), early stopping, dropout (NN), simpler model, cross-validation for hyperparameters, ensembling.

**Train / Validation / Test & Cross-Validation**
- Train learns parameters, validation tunes hyperparameters, test is touched only once for the final generalization estimate.
- k-fold CV: more reliable estimate when data is limited.
- ⚠️ **Data leakage**: any use of test/future information (e.g. scaling on the full dataset, using future features) inflates results — a very common trap question.

## B. Data & Feature Engineering

- **Feature scaling**: standardization (z-score) vs normalization (min-max). Matters for distance-based models (kNN, SVM, k-means) and gradient descent; tree models don't need it.
- **Categorical encoding**: one-hot, label encoding, target/mean encoding (watch leakage), embeddings.
- **Missing values**: drop, mean/median imputation, model-based imputation, add a "missingness" indicator.
- **Imbalanced data (must-know)**: resampling (oversample SMOTE / undersample), class weights, threshold tuning, and evaluate with **PR-AUC, not accuracy**.
- **Feature selection**: filter (correlation/mutual info), wrapper, embedded (L1, tree importance).

## C. Classic Supervised Models

**Linear Regression** — predicts a continuous value, minimizes MSE. Assumptions: linearity, i.i.d. normal errors, homoscedasticity. Normal equation vs gradient descent.

**Logistic Regression (very high frequency)**
- Classification (despite the name). Sigmoid maps a linear output to a probability in (0,1).
- Loss is cross-entropy (log loss), which is convex.
- Why not MSE for classification? → with sigmoid it becomes non-convex and suffers vanishing gradients.
- Interpretable: coefficients = change in log-odds.

**Regularization: L1 vs L2 (must-know)**
- L2 (Ridge): penalizes squared weights, shrinks them toward (but not to) zero, handles collinearity.
- L1 (Lasso): penalizes absolute weights, can drive weights **exactly to zero** → feature selection (sparsity).
- Elastic Net: L1 + L2.
- Geometric intuition: L1's diamond constraint has corners on the axes → sparse solutions.

**Decision Trees** — split on features using information gain (entropy) or Gini impurity. Pros: interpretable, no scaling needed, captures non-linearity. Cons: overfits easily, unstable.

**Random Forest** — bagging (bootstrap sampling) + random feature subsets, trees vote. Reduces variance, resists overfitting. Random features **decorrelate** the trees, which is what makes the ensemble work.

**Gradient Boosting / XGBoost / LightGBM (high frequency)**
- Boosting: sequential; each new tree fits the residual (negative gradient) of the previous ones, progressively reducing bias.
- vs Random Forest: RF is parallel and reduces variance; GBDT is sequential and reduces bias — usually higher accuracy but more prone to overfitting and needs tuning.
- XGBoost engineering: regularization term, 2nd-order Taylor approximation, parallelism, handles missing values. The king of tabular data.

**SVM** — finds the maximum-margin hyperplane. The kernel trick maps data to higher dimensions for non-linear classification (RBF kernel most common). Support vectors are the points that define the boundary.

**kNN** — lazy learner, votes among the k nearest neighbors. Needs scaling, sensitive to the curse of dimensionality, slow at inference.

**Naive Bayes** — Bayes' theorem + conditional independence assumption. Fast and effective for text classification.

## D. Unsupervised Learning (your strength — review well)

**k-means** — minimizes within-cluster sum of squares; iterates assign→update centroids. Needs preset k, assumes spherical clusters, sensitive to initialization (k-means++) and outliers. Choose k with the elbow method or silhouette score.

**DBSCAN** — density-based clustering; no preset cluster count, finds arbitrary shapes, labels noise. Parameters: `eps` and `min_samples`. Struggles with clusters of varying density.

**GMM + EM** — soft clustering with probabilities. EM alternates the E-step (posterior "responsibilities") and M-step (update means/covariances/weights); likelihood increases monotonically but can get stuck in local optima. k-means is a special case of GMM (spherical covariance, hard assignment).

**PCA (dimensionality reduction, must-know)** — finds orthogonal directions of maximum variance (principal components) for linear dimensionality reduction; mathematically an eigendecomposition of the covariance matrix / SVD. Uses: reduce dimensions, decorrelate, visualize, denoise. ⚠️ Standardize first. vs t-SNE/UMAP: those are non-linear and preserve local structure for visualization — don't use them as features for downstream models.

## E. Model Evaluation (must-know — answer by scenario)

**Classification metrics**
- Confusion matrix: TP / FP / TN / FN.
- Precision = TP/(TP+FP): of predicted positives, how many are real (care about false alarms, e.g. spam).
- Recall = TP/(TP+FN): of actual positives, how many were caught (care about misses, e.g. cancer screening, fraud).
- F1 = harmonic mean of precision & recall; meaningful when imbalanced.
- ROC-AUC: TPR vs FPR across thresholds, threshold-independent. **With heavy class imbalance, PR-AUC is more informative.**
- Accuracy trap: with 99% negatives, predicting all-negative still scores 99%.

**Regression metrics**: MSE / RMSE (sensitive to large errors), MAE (more robust to outliers), R² (fraction of variance explained).

## F. Optimization

**Gradient descent family**: Batch / Mini-batch / SGD — full-batch is stable but slow / mini-batch is the common compromise / SGD is noisy per sample. Learning rate: too high diverges, too low is slow — use schedules / warmup. Momentum, RMSprop, **Adam** (momentum + adaptive LR, the deep-learning default).

**Convex vs non-convex**: logistic/linear/SVM are convex with a global optimum; neural nets are non-convex — SGD finds a "good enough" local optimum.

---

## G. Deep Learning

**Neural network basics** — a neuron = weighted sum + activation; stacked layers = multilayer perceptron (MLP). **Universal approximation theorem**: a wide-enough single hidden layer can approximate any continuous function (doesn't mean it's easy to train).

**Backpropagation (must-know)** — uses the chain rule to compute gradients layer by layer from output back to input, then gradient descent updates weights. Forward pass computes loss; backward pass computes gradients.

**Activation functions**
- Sigmoid / tanh: saturate → vanishing gradients.
- **ReLU**: max(0,x) — simple, mitigates vanishing gradients, the default for deep nets. Variants: Leaky ReLU, GELU (common in Transformers).
- Output layer: linear for regression, sigmoid for binary, softmax for multiclass.

**Loss functions** — regression: MSE; classification: cross-entropy. Softmax + cross-entropy is the multiclass standard.

**Vanishing / exploding gradients** — from multiplying small/large gradients across deep nets. Fixes: ReLU, good initialization (He/Xavier), Batch Norm, residual connections (ResNet), gradient clipping.

**Regularization (deep-learning-specific)**
- **Dropout**: randomly drop neurons during training to prevent co-adaptation; acts like an ensemble.
- **Batch Normalization**: normalizes each layer's inputs → faster convergence, allows larger learning rates, mild regularization.
- Also: data augmentation, early stopping, weight decay.

**Optimizers**: Adam is the default; SGD with momentum is also used for large-scale / better generalization.

## H. CNNs (relevant to your CV background)

- **Convolution**: a small kernel slides over the image extracting local features; weight sharing → few parameters, translation invariance.
- **Pooling**: downsampling (max/avg) → less compute, larger receptive field, robustness to small shifts.
- Typical stack: conv → activation → pool, repeated, then fully connected. Classic nets: LeNet, AlexNet, VGG, ResNet (residual connections fix degradation in deep nets).
- Uses: image classification, detection (YOLO / Faster R-CNN), segmentation (U-Net) — ties to your point-cloud/imaging work.

## I. Sequence Models

- **RNN**: processes sequences, passes a hidden state as memory. Problem: vanishing gradients over long dependencies.
- **LSTM / GRU**: gating mechanisms decide what to remember/forget → handle long dependencies.
- Mostly replaced by Transformers now, but the concepts still come up.

## J. Transformers & Attention (modern focus)

- **Self-attention**: each token attends to all others, aggregating them weighted by relevance. Core formula: softmax(QKᵀ/√d)·V.
- **Why it replaced RNNs**: parallelizable (no time-step loop) and strong at long dependencies.
- **Transformer block**: multi-head attention + feed-forward + residual + LayerNorm + positional encoding.
- **Embeddings**: map tokens/items/users to dense vectors where similar things are close — connects to recommendation and vector search.

## K. LLMs / GenAI

- **Pretrain + fine-tune paradigm**: self-supervised pretraining (predict next token) → downstream fine-tuning / instruction tuning / RLHF.
- **Parameter-efficient fine-tuning (PEFT)**: LoRA etc. — train few parameters, save compute.
- **RAG (retrieval-augmented generation)**: retrieve external knowledge via embeddings and feed it to the LLM → reduces hallucination, keeps knowledge updatable. The mainstream enterprise LLM pattern.
- **Agents / function calling**: LLM calls tools, makes multi-step decisions.
- **Evaluation problem**: generative tasks have no single correct answer → human eval / LLM-as-judge / task-level metrics.

---

# Part 2B — Resume-Driven Deep Dives

> These map directly to what interviewers will dig into from your resume (fraud/LLM at Amazon, DS/experimentation at IBM, RAG/agents at AceRocket). Same rule: essence in 1–2 lines + when to use.

## L. Imbalanced, Cost-Sensitive Learning & Calibration (fraud / risk)

- **Beyond resampling**: SMOTE / undersampling / class weights are table stakes. In fraud also use **cost-sensitive learning** (a cost matrix — a missed fraud costs far more than a false alarm) and pick the **operating threshold** from the business cost curve, not 0.5.
- **Focal loss**: down-weights easy negatives so training focuses on the hard/rare positives — useful when positives are <1%.
- **Probability calibration (must for thresholding)**: raw scores aren't true probabilities. **Platt scaling** (sigmoid fit) or **isotonic regression** calibrate them; check with a reliability diagram / ECE. You need calibrated scores to hit a target precision/recall.
- **Evaluate with PR-AUC** (and recall@fixed-precision), not accuracy or ROC-AUC, under heavy imbalance.
- **Fraud-specific realities**: labels arrive **late** (chargebacks come weeks later) → label delay + feedback loops; fraudsters adapt → **adversarial concept drift** → models decay fast, need frequent monitoring/retraining.

## M. Multi-Task Learning (MTL)

- **Idea**: one network learns several related tasks via a **shared backbone** + **task-specific heads** (shared embeddings, per-task output layers).
- **Why**: shared representation regularizes, improves generalization, and cuts training/serving cost vs one model per task.
- **When it helps**: tasks share signal; it hurts when tasks conflict → **negative transfer**.
- **Hard vs soft parameter sharing**; **loss weighting** matters (fixed, or uncertainty-based) so one task doesn't dominate.

## N. LLM Fine-Tuning (PEFT) & Inference Optimization

- **LoRA**: freeze base weights, learn a **low-rank update** ΔW = B·A (rank r ≪ d) on chosen matrices; train only A,B → tiny % of params, small memory. Knobs: **rank r**, **alpha** (scaling), which layers (usually attention q/v).
- **QLoRA**: load the base model in **4-bit (NF4)** quantization, train LoRA adapters on top → fine-tune a big model on one GPU.
- **Quantization**: INT8/INT4 to shrink memory and speed inference. **PTQ** (post-training: GPTQ, AWQ) vs **QAT** (quantization-aware, higher accuracy). Trade a little accuracy for latency/cost.
- **Other PEFT**: prefix / prompt tuning, adapters.
- **Inference optimization (ties to "sub-100ms")**: quantization, **knowledge distillation** (small student mimics big teacher), **batching** / continuous batching, KV-cache, runtimes (ONNX Runtime, TensorRT, vLLM), response caching.

## O. MLOps & the Production ML Lifecycle

- **Lifecycle**: data → features → train → offline eval → deploy → monitor → iterate. The hard part is everything after "train."
- **Drift**: **data drift** (input distribution shifts) vs **concept drift** (X→y relationship shifts). Detect with **PSI**, **KS test**, KL divergence on features/scores; watch live metrics for degradation.
- **Monitoring**: data quality, prediction distribution, latency, and (once labels land) live metrics + alerts → trigger retraining.
- **Feature store**: one source of features for train and serve to avoid **training-serving skew**; online (low-latency) vs offline (batch).
- **Deployment safety**: **shadow** (run new model silently) → **canary** / gradual rollout → **A/B** → **rollback**; a **model registry** for versioning.
- **CI/CD for ML**: automated retraining pipelines, reproducibility, data + model versioning.

## P. Retrieval, Vector Search & RAG (incl. Elasticsearch / hybrid)

- **Lexical search**: **inverted index** + **BM25** (TF-IDF-style scoring) — exact-term matching, what classic **Elasticsearch** does. Fast, great for keywords/rare tokens, no semantics.
- **Dense / semantic search**: embed query & docs, retrieve by cosine/dot similarity. At scale needs **ANN** — **HNSW** (graph) or **IVF/PQ**; stores: FAISS, pgvector, Elasticsearch `dense_vector`, Pinecone.
- **Hybrid search (best in practice)**: combine BM25 + dense, fuse with **RRF (reciprocal rank fusion)**, then **rerank** top-k with a **cross-encoder** (slower, accurate).
- **RAG pipeline**: chunk → embed → retrieve (top-k, **MMR** for diversity) → rerank → stuff into context → generate. Reduces hallucination, keeps knowledge updatable.
- **RAG evaluation**: retrieval hit-rate / recall@k, and answer **faithfulness / groundedness** (LLM-as-judge). Failure modes: bad chunking, embedding mismatch, lost-in-the-middle.

## Q. LLM Agents

- **Definition**: LLM as controller + **tool/function calling** + multi-step loop + state. Test "is it an agent?" by four abilities: call tools, see the last result, decide the next step, keep state.
- **Patterns**: **ReAct** (reason + act loop), planning (decompose a goal), reflection/self-critique, **multi-agent** (lead–worker, debate).
- **Tooling**: function calling, **MCP (Model Context Protocol)** — a standard way to connect models to tools/data; orchestration (LangChain, LangGraph, LlamaIndex, Agents SDK).
- **Hard parts**: reliability (errors cascade), termination (knowing when to stop), cost/latency (multi-agent can burn ~15× tokens), evaluation.

## R. Experimentation, A/B Testing & Causal Inference (DS core)

- **Hypothesis testing**: null vs alternative, **p-value**, **Type I** (false positive, α) vs **Type II** (false negative, β), **power** = 1−β. Multiple-testing correction (Bonferroni / FDR).
- **A/B design**: pick metric + **MDE** (minimum detectable effect) → compute **sample size** → randomize → run to a pre-set duration. Watch **novelty/primacy effects**, peeking (use **sequential testing** if you must peek), network effects.
- **When you can't randomize → causal inference**: confounders, **difference-in-differences (DiD)**, **propensity-score** matching, **instrumental variables**, regression discontinuity.
- **Bayesian view**: priors + posteriors; Bayesian A/B gives "P(B beats A)" and credible intervals.

## S. Recommendation Systems

- **Collaborative filtering**: from user–item interactions. **Matrix factorization** (learn user/item embeddings, dot product = affinity). vs **content-based** (item/user features). **Hybrid** combines both.
- **Cold start** (new user/item) → fall back to content / popularity.
- **Implicit feedback** (clicks/views, not ratings) is the common case → different losses (BPR, weighted).
- **Two-tower model**: separate user & item encoders → embeddings → ANN retrieval, then a heavier ranker (retrieval + ranking = the modern industrial pattern).
- **Metrics**: recall@k, **NDCG**, MAP, hit-rate — ranking-aware, not accuracy.

## T. Time Series (brief)

- **Components**: trend, seasonality, autocorrelation. Make it stationary (differencing) before classic models.
- **Classic**: ARIMA / SARIMA, exponential smoothing. **Features for ML**: lags, rolling stats, calendar/seasonal features.
- **Validation**: **time-based split** / rolling-origin CV — never shuffle, never use the future.

## U. Interpretability & Misc Fundamentals

- **Interpretability**: **SHAP** (Shapley-value attributions, consistent) and **LIME** (local linear surrogate) — needed for fraud/regulated models where you must justify a decision.
- **Ensembling beyond bagging/boosting**: **stacking** (a meta-model learns to combine base models).
- **Cross-validation variants**: **stratified** (keep class ratios), **group** (no leakage across a user/group), **time-series** (rolling).
- **Loss functions to know**: focal (imbalance), **Huber** (robust regression), hinge (SVM).

---

## ✅ Pre-Interview Self-Check (if you can answer on the spot, you're ready)

- [ ] Bias-variance in one line + how to diagnose overfitting
- [ ] L1 vs L2 + why L1 is sparse
- [ ] Why logistic regression uses cross-entropy, not MSE
- [ ] Random forest vs gradient boosting — the essential difference
- [ ] When to use precision / recall / F1 / AUC
- [ ] DBSCAN vs k-means vs GMM — when each applies
- [ ] Backprop in one line + how to fix vanishing gradients
- [ ] What Dropout and BatchNorm each do
- [ ] Self-attention formula + why it replaced RNNs
- [ ] What RAG is and what problem it solves
- [ ] LoRA / QLoRA + quantization — how they save memory, PTQ vs QAT
- [ ] Multi-task learning — shared backbone + heads, negative transfer
- [ ] Data drift vs concept drift + how to detect (PSI / KS)
- [ ] Feature store & training-serving skew; shadow / canary / rollback
- [ ] BM25 vs dense vs hybrid search + RRF + cross-encoder rerank
- [ ] Full RAG pipeline + how to evaluate retrieval and faithfulness
- [ ] What makes an LLM an "agent" + ReAct + its failure modes
- [ ] A/B test design (MDE, sample size, peeking) + Type I/II & power
- [ ] Causal inference when you can't randomize (DiD, propensity)
- [ ] Recsys: matrix factorization, cold start, two-tower, NDCG
- [ ] Probability calibration + why you need it to set a threshold
- [ ] SHAP / LIME — when interpretability matters
- [ ] Every resume item survives "what → why that choice → how evaluated"

---

*Sources: your homepage (dongzhang84.github.io), GitHub, LinkedIn, Notion job-tracker; ML content is general interview scope. Fill in the Amazon / IBM / AceRocket sections and I'll tighten Part 1.*
