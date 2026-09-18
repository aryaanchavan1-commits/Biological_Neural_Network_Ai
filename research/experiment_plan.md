# BIO-NN Experiment Plan

## Overview

This document defines the complete experimental plan for the BIO-NN research framework, organized into 6 phases over 12 weeks. Each experiment specifies the hypothesis being tested, configuration, datasets, metrics, expected results, baselines, ablations, and statistical analysis.

**Total Experiments:** 24
**Total Runs (estimated):** ~1,800 (including seeds and ablations)
**Compute Budget:** ~200 GPU-hours (estimated on A100)

---

## Phase 1: Foundation (Weeks 1-2)

**Goal:** Validate core SNN components and establish baselines.

### E1.1: LIF Neuron Validation

**Hypothesis Being Tested:** The LIF neuron implementation correctly models membrane potential dynamics and spike generation.

**Configuration:**
- Neuron model: Leaky Integrate-and-Fire (LIF)
- Parameters: τ_m = 20ms, V_rest = -65mV, V_th = -50mV, V_reset = -65mV, R_m = 10MΩ
- Input: constant current injection (0.5nA, 1.0nA, 1.5nA)
- Duration: 1000ms per trial
- Timestep: 1ms (1000 steps)
- Trials: 100 per current level

**Datasets:** Synthetic (current injection protocols)

**Metrics:**
- Membrane potential trajectory (vs. analytical solution)
- Spike timing (vs. analytical solution)
- Firing rate (vs. analytical f-I curve)
- Maximum absolute error in membrane potential
- Spike timing error (ms)

**Expected Results:**
- Membrane potential error < 0.1mV at all timesteps
- Spike timing error < 1ms (within one timestep)
- Firing rate within 5% of analytical prediction
- Refractory period correctly implemented (2ms absolute, 5ms relative)

**Baselines:**
- Analytical LIF solution (exact mathematical model)
- Brian2 simulator output (reference implementation)
- Norse library LIF implementation

**Ablations:**
- Test with different τ_m values (10ms, 20ms, 50ms)
- Test with different V_th values (-55mV, -50mV, -45mV)
- Test with different timestep sizes (0.5ms, 1ms, 2ms, 5ms)

**Statistical Analysis:**
- Descriptive statistics: mean, std, max of error across trials
- No inferential statistics needed (deterministic comparison)
- Report: error distributions and convergence analysis

---

### E1.2: Basic SNN on MNIST (Baseline)

**Hypothesis Being Tested:** A standard SNN can achieve competitive accuracy on MNIST, establishing a baseline for subsequent experiments.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Neuron model: LIF (from E1.1)
- Encoding: rate coding (50 timesteps)
- Training: surrogate gradient (fast sigmoid)
- Optimizer: Adam, lr=1e-3
- Epochs: 20
- Batch size: 128
- Sparsity: dense (100% connectivity)

**Datasets:**
- MNIST (60k train, 10k test)
- Train/test split: 50k/10k (standard)

**Metrics:**
- Test accuracy (%)
- Training loss (cross-entropy)
- Spike count per sample (average)
- Firing rate per layer
- Training time (wall-clock)
- Convergence speed (epochs to 95% of final accuracy)

**Expected Results:**
- Test accuracy: 97.5-98.5%
- Average spike count: 100-200 per sample
- Firing rate: 10-30% per layer
- Training time: < 5 minutes on A100

**Baselines:**
- Standard MLP (same architecture, ReLU, no spiking)
- SNN with 100 timesteps
- SNN with 200 timesteps
- State-of-the-art SNN results on MNIST

**Ablations:**
- Timestep count: 10, 20, 50, 100, 200
- Hidden layer size: 128, 256, 512, 1024
- Surrogate gradient function: fast sigmoid, polynomial, triangular

**Statistical Analysis:**
- Report mean ± std over 10 seeds
- Compare against MLP baseline: paired t-test
- Report confidence intervals for accuracy
- Learning curve comparison (accuracy vs. epoch)

---

### E1.3: Rate Coding vs. Temporal Coding

**Hypothesis Being Tested:** Temporal coding (spike timing) provides better accuracy-efficiency trade-off than rate coding for MNIST classification.

**Configuration:**
- **Rate Coding:** 50 timesteps, spike probability proportional to input intensity
- **Temporal Coding:** Latency coding, spike time inversely proportional to input intensity (early = strong)
- **Population Coding:** Distributed representation across neuron populations
- Architecture: 784 → 256 → 128 → 10 (same for all)
- Training: surrogate gradient
- All other params: same as E1.2

**Datasets:** MNIST

**Metrics:**
- Test accuracy (%)
- Total spike count per sample
- Classification latency (timesteps to first correct prediction)
- Energy efficiency: accuracy / spike_count
- Temporal sparsity: fraction of timesteps with no spikes

**Expected Results:**
- Temporal coding: 97-98% accuracy with 30-50% fewer spikes
- Rate coding: 97.5-98.5% accuracy (baseline)
- Temporal coding achieves 95% accuracy within first 10 timesteps
- Population coding: slightly higher accuracy but more spikes

**Baselines:**
- Rate coding (E1.2 results)
- Temporal coding with different encodings (latency, rank-order, phase)
- XAX encoding comparison from literature

**Ablations:**
- Number of timesteps: 10, 20, 50, 100
- Encoding resolution: 4-bit, 8-bit, 16-bit temporal precision
- Mixed coding: rate for early layers, temporal for later layers

**Statistical Analysis:**
- Paired t-test: rate vs. temporal coding accuracy
- Efficiency comparison: accuracy/spike_count ratio
- ROC curve for early classification (accuracy vs. latency trade-off)
- Report: Pareto frontier of accuracy vs. efficiency

---

### E1.4: Surrogate Gradient vs. STDP Learning

**Hypothesis Being Tested:** Surrogate gradient learning achieves faster convergence and higher accuracy than STDP for supervised SNN training.

**Configuration:**
- **Surrogate Gradient:**
  - Fast sigmoid surrogate: σ'(x) = 1/(1+|x|)²
  - Learning rate: 1e-3
  - Optimizer: Adam
  - Batch size: 128
- **STDP:**
  - Learning rate: 0.01 (pre-synaptic), 0.005 (post-synaptic)
  - Weight bounds: [0, w_max] where w_max = 1.0
  - Reward modulated: yes (R-STDP)
  - Batch size: 1 (online learning)
- **Hybrid:** STDP for feature extraction, surrogate gradient for classification
- Architecture: 784 → 256 → 128 → 10
- All methods: 50 timesteps, rate coding

**Datasets:** MNIST

**Metrics:**
- Test accuracy (%)
- Training convergence speed (epochs to target accuracy)
- Weight distribution analysis (mean, std, sparsity)
- Training time (wall-clock)
- Biological plausibility score (subjective assessment)

**Expected Results:**
- Surrogate gradient: 97.5-98.5% accuracy, fast convergence (5-10 epochs)
- STDP: 90-95% accuracy, slower convergence (50-100 epochs)
- Hybrid: 96-97% accuracy, moderate convergence
- Surrogate gradient is 5-10x faster in wall-clock time

**Baselines:**
- Backpropagation on rate-coded network (non-spiking)
- ANN-to-SNN conversion
- Other surrogate functions: triangular, Gaussian

**Ablations:**
- Surrogate functions: fast sigmoid, polynomial, triangular, Gaussian
- STDP variants: symmetric, asymmetric, triplet STDP
- Learning rate schedules for both methods
- Network size: small (128), medium (256), large (512)

**Statistical Analysis:**
- Learning curve comparison: accuracy vs. epoch with confidence bands
- Convergence speed: time to reach 95% of final accuracy
- Final accuracy comparison: ANOVA across methods
- Computational cost: time per epoch comparison

---

## Phase 2: Plasticity (Weeks 3-4)

**Goal:** Validate individual plasticity mechanisms and measure their contributions.

### E2.1: Hebbian Learning Alone

**Hypothesis Being Tested:** Hebbian learning can extract useful features from MNIST without supervised labels.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Learning rule: Hebbian (Δw = η × pre × post)
- Weight normalization: divisive normalization
- Weight bounds: [0, 1.0]
- Unsupervised pre-training: 20 epochs
- Supervised fine-tuning: 10 epochs (last layer only)
- Sparsity constraint: top-k inhibition (k=10% of neurons)

**Datasets:** MNIST

**Metrics:**
- Unsupervised feature quality (visualize learned filters)
- Clustering accuracy (k-means on learned features)
- Linear classifier accuracy on learned features
- Weight sparsity (% of weights near zero)
- Feature diversity (cosine similarity between filters)

**Expected Results:**
- Unsupervised clustering accuracy: 60-70%
- Linear classifier accuracy: 85-90%
- Learned filters resemble Gabor-like edge detectors
- Weight sparsity: 30-50%

**Baselines:**
- Random features + linear classifier
- PCA features + linear classifier
- Autoencoder features + linear classifier
- Sparse autoencoder features

**Ablations:**
- Learning rate: 0.001, 0.01, 0.1
- Sparsity level: 5%, 10%, 20%, 50%
- Normalization: none, divisive, subtractive
- Weight bounds: [0,1], [-1,1], unbounded

**Statistical Analysis:**
- Feature quality: visualize top-50 filters per condition
- Accuracy comparison: ANOVA across methods
- Report: feature diversity metrics and weight distributions

---

### E2.2: STDP Alone

**Hypothesis Being Tested:** STDP can learn temporal features and improve sequence discrimination.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Learning rule: asymmetric STDP (τ+ = 20ms, τ- = 20ms)
- Weight initialization: random uniform [0, 0.5]
- Training: online STDP (batch size 1)
- Post-processing: linear readout trained separately
- Epochs: 50 (STDP), 10 (readout)

**Datasets:**
- MNIST (static)
- Temporal MNIST (sequential pixel presentation)
- DVS Gesture dataset (event-based)

**Metrics:**
- Readout accuracy (%)
- Temporal selectivity of neurons
- STDP weight change distribution
- Spike timing correlations
- Receptive field quality

**Expected Results:**
- Static MNIST: 80-85% accuracy (STDP is weak for static data)
- Temporal MNIST: 88-92% accuracy (STDP benefits from temporal structure)
- DVS Gesture: 85-90% accuracy (event-based data suits STDP)
- Temporal selectivity: > 30% of neurons show timing preference

**Baselines:**
- Rate-coded Hebbian (E2.1)
- Random features + readout
- Convolutional SNN trained with surrogate gradient
- ANN baseline on same data

**Ablations:**
- STDP time constants: τ = 10, 20, 50ms
- Symmetric vs. asymmetric STDP
- Triplet STDP vs. pairwise STDP
- Weight bounds: [0, 0.5], [0, 1.0], unbounded

**Statistical Analysis:**
- Accuracy comparison across datasets and methods
- Temporal selectivity: circular statistics for spike timing
- Weight distribution analysis: KS test for different conditions
- Report: temporal receptive fields and spike rasters

---

### E2.3: Reward-Modulated STDP (R-STDP)

**Hypothesis Being Tested:** R-STDP can learn classification-relevant features through reward signals, combining unsupervised feature learning with supervised guidance.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Learning rule: R-STDP with eligibility traces
- Reward signal: correct/incorrect classification (binary)
- Eligibility trace decay: τ_e = 500ms
- Learning rate: 0.01
- Training: batch size 32, reward delayed by 100ms
- Pre-training: unsupervised STDP (10 epochs)
- R-STDP training: 20 epochs

**Datasets:** MNIST, Fashion-MNIST

**Metrics:**
- Classification accuracy (%)
- Reward correlation: weight change vs. reward signal
- Feature alignment: learned features vs. class-discriminative features
- Learning speed: epochs to 90% accuracy
- Stability: accuracy variance across training

**Expected Results:**
- MNIST: 90-93% accuracy
- Fashion-MNIST: 82-86% accuracy
- Weight changes positively correlated with reward (r > 0.3)
- 2-3x faster convergence than pure STDP

**Baselines:**
- Pure STDP (E2.2)
- Surrogate gradient (E1.4)
- REINFORCE algorithm
- TD-learning based SNN

**Ablations:**
- Eligibility trace decay: 100ms, 500ms, 1000ms
- Reward granularity: per-sample, per-batch, per-epoch
- Reward type: binary, confidence-based, hierarchical
- Pre-training duration: 0, 5, 10, 20 epochs

**Statistical Analysis:**
- Learning curve comparison with confidence bands
- Weight-reward correlation: Pearson r with significance test
- Feature alignment: cosine similarity with class-discriminative directions
- Report: reward signal time course and weight updates

---

### E2.4: Homeostatic Plasticity

**Hypothesis Being Tested:** Homeostatic plasticity (synaptic scaling and intrinsic excitability) stabilizes network activity and prevents runaway excitation/inhibition during training.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Homeostatic mechanisms:
  - Synaptic scaling: target firing rate = 10Hz, τ_scale = 1000ms
  - Intrinsic plasticity: target membrane potential = -55mV, τ_ip = 500ms
- Training: surrogate gradient with homeostatic regularization
- Homeostatic update frequency: every 100ms (100 timesteps)
- Baseline: no homeostatic plasticity

**Datasets:** MNIST

**Metrics:**
- Firing rate distribution (mean, std, across layers)
- Firing rate stability over training (coefficient of variation)
- Weight distribution evolution
- Training stability (loss variance)
- Final accuracy (%)
- Time to convergence

**Expected Results:**
- Firing rate mean: 10-15Hz (near target)
- Firing rate std: < 50% of mean (stable activity)
- Training loss variance: 50% lower than baseline
- Accuracy: 97-98% (maintained or slightly improved)
- Convergence: 10-20% faster than baseline

**Baselines:**
- No homeostatic plasticity
- Only synaptic scaling
- Only intrinsic excitability
- Batch normalization (artificial equivalent)
- Layer normalization

**Ablations:**
- Target firing rate: 5Hz, 10Hz, 20Hz, 50Hz
- Homeostatic time constant: 100ms, 500ms, 1000ms, 5000ms
- Update frequency: 10ms, 100ms, 1000ms
- Mechanism combination: scaling only, IP only, both

**Statistical Analysis:**
- Firing rate stability: CV comparison across conditions
- Weight distribution: KS test for distribution differences
- Training stability: loss variance comparison (F-test)
- Convergence speed: time to reach 95% of final accuracy
- Report: firing rate time courses and weight histograms

---

### E2.5: Plasticity Ablation Study

**Hypothesis Being Tested:** Each plasticity mechanism (Hebbian, STDP, R-STDP, homeostatic) contributes independently to learning, with potential synergy between mechanisms.

**Configuration:**
- 8 conditions (full factorial):
  1. No plasticity (baseline SNN)
  2. Hebbian only
  3. STDP only
  4. R-STDP only
  5. Homeostatic only
  6. Hebbian + Homeostatic
  7. STDP + Homeostatic
  8. All mechanisms (full BIO-NN)
- Architecture: 784 → 256 → 128 → 10
- Training: same protocol for all conditions
- Seeds: 10 per condition

**Datasets:** MNIST, Fashion-MNIST

**Metrics:**
- Test accuracy (%)
- Training loss curve
- Firing rate statistics
- Weight sparsity
- Feature quality (visual inspection)
- Learning speed (epochs to target)

**Expected Results:**
- No plasticity: 97.5% (surrogate gradient baseline)
- Hebbian only: 85-90% (unsupervised features)
- STDP only: 80-85% (temporal features)
- R-STDP only: 90-93% (reward-guided)
- Homeostatic only: 97-98% (stability only)
- Combinations: 93-97% (approaching full model)
- All mechanisms: 97-98% (target)

**Baselines:**
- Standard MLP (ReLU, SGD)
- SNN with surrogate gradient only
- SNN with data augmentation

**Ablations:**
- Each mechanism on/off (8 conditions)
- Mechanism interaction: 2-way and 3-way interactions
- Learning rate sensitivity per mechanism
- Training duration sensitivity per mechanism

**Statistical Analysis:**
- Factorial ANOVA: mechanism × condition
- Main effects and interaction effects (Type III SS)
- Pairwise comparisons with Bonferroni correction
- Effect size: η² for each mechanism and interaction
- Report: effect size table with confidence intervals

---

## Phase 3: Structure (Weeks 5-6)

**Goal:** Validate structural plasticity mechanisms (pruning, growth).

### E3.1: Static Sparse Connectivity vs. Dense

**Hypothesis Being Tested:** Static sparse connectivity can achieve competitive accuracy with dense connectivity while reducing computation.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- Connectivity levels:
  - Dense: 100%
  - 50% sparse
  - 25% sparse
  - 10% sparse
  - 5% sparse
- Sparse initialization: random mask (fixed during training)
- Training: surrogate gradient, Adam optimizer
- All other params: same as E1.2

**Datasets:** MNIST

**Metrics:**
- Test accuracy (%)
- Spike count per sample
- Active synapses per forward pass
- Training time
- Inference latency
- Parameter efficiency: accuracy / parameters

**Expected Results:**
- Dense: 98% accuracy
- 50% sparse: 97.5-98% (minimal loss)
- 25% sparse: 96-97% (small loss)
- 10% sparse: 93-95% (noticeable loss)
- 5% sparse: 88-92% (significant loss)
- Spike count: 20-50% reduction at 25% sparsity

**Baselines:**
- Dense SNN (100% connectivity)
- Structured pruning (channel-wise)
- Knowledge distillation from dense to sparse

**Ablations:**
- Random vs. structured sparsity
- Layer-wise sparsity (more sparse in early layers)
- Sparsity scheduling (gradual sparsification during training)
- Different random seeds for connectivity masks

**Statistical Analysis:**
- Accuracy vs. sparsity curve with confidence bands
- Efficiency metrics: accuracy/spike_count ratio
- Correlation: sparsity level vs. accuracy (Spearman ρ)
- Report: connectivity visualizations and spike statistics

---

### E3.2: Connection Pruning During Training

**Hypothesis Being Tested:** Dynamic pruning during training removes redundant connections while maintaining accuracy, resulting in a more efficient network.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10 (initially dense)
- Pruning methods:
  - Magnitude pruning: remove weights < threshold
  - Activity pruning: remove connections with low activity
  - Gradient pruning: remove connections with small gradients
  - Combined: magnitude + activity
- Pruning schedule:
  - Gradual: 10% per epoch (epochs 5-15)
  - Aggressive: 50% at epoch 10
  - Adaptive: prune when activity < 1%
- Target sparsity: 50%, 75%, 90%

**Datasets:** MNIST

**Metrics:**
- Test accuracy (%)
- Sparsity over training (time course)
- Weight distribution evolution
- Connection importance: correlation with gradient magnitude
- Recovery accuracy: retrain after pruning
- Pruning speed: time to prune

**Expected Results:**
- Gradual pruning: 97-98% accuracy at 50% sparsity
- Aggressive pruning: 95-97% accuracy at 50% sparsity
- Adaptive pruning: 97-98% accuracy at 40-60% sparsity
- Recovery: retraining restores < 1% accuracy loss
- Gradient pruning outperforms magnitude pruning

**Baselines:**
- Magnitude pruning (standard approach)
- Movement pruning (huggingface)
- Structured pruning (channel/layer-wise)
- Lottery ticket hypothesis (rewind to init)

**Ablations:**
- Pruning schedule: gradual, aggressive, adaptive
- Pruning criterion: magnitude, activity, gradient, combined
- Target sparsity: 25%, 50%, 75%, 90%
- Rewinding: no rewind, rewind to epoch 0, rewind to epoch 5

**Statistical Analysis:**
- Accuracy vs. sparsity curve for each method
- Statistical comparison of methods at each sparsity level
- Sparsity time course visualization
- Weight importance analysis: which weights are pruned?

---

### E3.3: Connection Growth During Training

**Hypothesis Being Tested:** Dynamic growth of new connections during training allows the network to adapt to task demands and increase capacity where needed.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10 (initially sparse, 25% connectivity)
- Growth methods:
  - Activity-based: grow connections to highly active neuron pairs
  - Gradient-based: grow connections with high gradient magnitude
  - Random: grow random connections (control)
  - Hebbian: grow connections where pre and post are co-active
- Growth triggers:
  - When neuron firing rate drops below threshold (1Hz)
  - When loss plateaus (no improvement for 5 epochs)
  - Every N epochs (periodic growth)
- Growth rate: 5% of current connections per growth event
- Maximum connectivity: 100%

**Datasets:** MNIST

**Metrics:**
- Test accuracy (%)
- Connectivity over training (time course)
- New connection utility: correlation with accuracy improvement
- Network size: total active connections
- Growth efficiency: accuracy gain per new connection
- Task-specific connectivity: connections unique to each task

**Expected Results:**
- Activity-based growth: 97-98% accuracy, 40-60% final connectivity
- Gradient-based growth: 97-98% accuracy, 35-55% final connectivity
- Random growth: 96-97% accuracy, 50-70% final connectivity
- Hebbian growth: 97-98% accuracy, 45-65% final connectivity
- Growth efficiency: > 0.1% accuracy per 1% new connections

**Baselines:**
- Static sparse (E3.1)
- Dynamic pruning only (E3.2)
- Neural architecture search (NAS)
- Network expansion (add neurons)

**Ablations:**
- Growth criterion: activity, gradient, random, Hebbian
- Growth trigger: threshold, plateau, periodic
- Growth rate: 1%, 5%, 10%, 20% per event
- Maximum connectivity: 50%, 75%, 100%

**Statistical Analysis:**
- Growth time course: connectivity vs. training epoch
- New connection utility: correlation with subsequent accuracy change
- Efficiency comparison: accuracy gain per connection across methods
- Report: connectivity visualizations and growth patterns

---

### E3.4: Combined Growth + Pruning

**Hypothesis Being Tested:** Combining growth and pruning allows the network to simultaneously remove redundant connections and add useful ones, achieving better efficiency-accuracy trade-off than either alone.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10 (initially 50% sparse)
- Combined strategies:
  - Grow first, prune later (grow 50%, then prune to 25%)
  - Prune first, grow later (prune to 10%, then grow to 25%)
  - Simultaneous: grow 5%, prune 5% per epoch
  - Adaptive: grow when needed, prune when redundant
- Growth criterion: activity-based
- Pruning criterion: magnitude-based
- Balance factor: growth rate = pruning rate (or adaptive)

**Datasets:** MNIST, Fashion-MNIST

**Metrics:**
- Test accuracy (%)
- Final connectivity (%)
- Net change: connections added - connections removed
- Stability: connectivity variance over training
- Efficiency: accuracy / final connectivity
- Dynamic range: max - min connectivity during training

**Expected Results:**
- Simultaneous: 97.5-98% accuracy, 25-35% final connectivity
- Adaptive: 97-98% accuracy, 20-30% final connectivity
- Grow then prune: 97-98% accuracy, 30-40% final connectivity
- Prune then grow: 96-97% accuracy, 25-35% final connectivity
- Dynamic range: 20-40% (network adapts capacity)

**Baselines:**
- Static sparse (E3.1)
- Pruning only (E3.2)
- Growth only (E3.3)
- Network surgery (manual adjustment)
- Neural architecture search

**Ablations:**
- Order of operations: grow-prune, prune-grow, simultaneous, adaptive
- Balance: equal rates, more growth, more pruning
- Timing: early, late, continuous
- Criteria: different combinations of growth/pruning rules

**Statistical Analysis:**
- Accuracy vs. connectivity for each strategy
- Dynamic range comparison across strategies
- Stability analysis: connectivity variance
- Efficiency metric: accuracy per unit connectivity
- Report: connectivity time courses and net change analysis

---

### E3.5: Structural Plasticity Ablation

**Hypothesis Being Tested:** Structural plasticity (growth + pruning) provides benefits beyond what can be achieved with synaptic plasticity alone.

**Configuration:**
- 6 conditions:
  1. No plasticity (baseline SNN)
  2. Synaptic plasticity only (STDP + homeostatic)
  3. Structural plasticity only (growth + pruning, no weight changes)
  4. Synaptic + structural (full BIO-NN)
  5. Synaptic only + static sparse connectivity
  6. Structural only + random weight initialization
- Architecture: 784 → 256 → 128 → 10
- Training: same protocol for all conditions
- Seeds: 10 per condition

**Datasets:** MNIST, Fashion-MNIST

**Metrics:**
- Test accuracy (%)
- Forgetting measure (if trained sequentially)
- Representation stability (RSA)
- Connectivity pattern analysis
- Weight distribution analysis
- Computational cost

**Expected Results:**
- No plasticity: 97.5% accuracy
- Synaptic only: 97-98% accuracy
- Structural only: 90-95% accuracy (limited by no learning rule)
- Synaptic + structural: 97-98% accuracy (with better efficiency)
- Synaptic + static sparse: 96-97% accuracy
- Structural + random: 85-90% accuracy

**Baselines:**
- Standard SNN (no plasticity)
- Pure synaptic plasticity (STDP + homeostatic)
- Dynamic sparse training (from literature)
- Network pruning + fine-tuning

**Ablations:**
- Each mechanism on/off (6 conditions)
- Growth rate vs. pruning rate ratio
- Structural plasticity timing: during training, after training, continuous
- Interaction between synaptic and structural plasticity

**Statistical Analysis:**
- ANOVA: plasticity type × structural vs. synaptic
- Pairwise comparisons with Tukey's HSD
- Effect size: η² for each mechanism
- Interaction effects: is there synergy?
- Report: mechanism contribution table with effect sizes

---

## Phase 4: Continual Learning (Weeks 7-8)

**Goal:** Test BIO-NN on continual learning benchmarks and compare against established methods.

### E4.1: Split MNIST (5 Tasks)

**Hypothesis Being Tested:** BIO-NN reduces catastrophic forgetting on split MNIST compared to standard baselines.

**Configuration:**
- Tasks: 5 binary classification tasks (0-1, 2-3, 4-5, 6-7, 8-9)
- Order: fixed (0-1 first, 8-9 last)
- Training: 5 epochs per task
- No task identifier during test (class-incremental setting)
- BIO-NN: full model with all mechanisms

**Datasets:** Split MNIST (5 tasks)

**Metrics:**
- Average accuracy (AA) across all tasks after final task
- Backward transfer (BWT)
- Forward transfer (FWT)
- Forgetting measure (FM)
- Per-task accuracy over time
- Model size (number of parameters)

**Expected Results:**
- BIO-NN: AA ≥ 95%, BWT ≥ -5%
- Fine-tuning: AA ≈ 80%, BWT ≈ -30%
- EWC: AA ≈ 88%, BWT ≈ -15%
- iCaRL: AA ≈ 90%, BWT ≈ -10%
- GEM: AA ≈ 92%, BWT ≈ -8%

**Baselines:**
- Fine-tuning (no continual learning method)
- EWC (λ = 1000, 5000, 10000)
- iCaRL (buffer size 200, 500, 1000)
- GEM (buffer size 200, 500, 1000)
- LwF (λ = 1, 5, 10)
- Standard SNN with fine-tuning

**Ablations:**
- Buffer size: 0, 200, 500, 1000, 2000
- Training epochs per task: 1, 5, 10, 20
- Task order: ascending, descending, random
- Network size: 128, 256, 512 hidden units

**Statistical Analysis:**
- One-way ANOVA across methods
- Post-hoc: Tukey's HSD pairwise comparisons
- Effect size: Cohen's d for BIO-NN vs. each baseline
- Learning curves: per-task accuracy over time
- Report: confusion matrices and forgetting analysis

---

### E4.2: Permuted MNIST

**Hypothesis Being Tested:** BIO-NN maintains accuracy across permuted MNIST tasks, demonstrating stability under distribution shifts.

**Configuration:**
- Tasks: 10 tasks with random pixel permutations
- Each task: different permutation (fixed per seed)
- Training: 5 epochs per task
- BIO-NN: full model with all mechanisms

**Datasets:** Permuted MNIST (10 tasks)

**Metrics:**
- Average accuracy (AA)
- Backward transfer (BWT)
- Permutation stability: accuracy on old permutations
- Representation overlap: RSA between task representations
- Computational cost

**Expected Results:**
- BIO-NN: AA ≥ 88%, BWT ≥ -8%
- Fine-tuning: AA ≈ 70%, BWT ≈ -25%
- EWC: AA ≈ 78%, BWT ≈ -15%
- iCaRL: AA ≈ 80%, BWT ≈ -12%

**Baselines:**
- Same as E4.1

**Ablations:**
- Number of tasks: 5, 10, 20, 50
- Permutation strength: partial (50%), full (100%)
- Task similarity: related vs. unrelated permutations

**Statistical Analysis:**
- Same as E4.1
- Additional: permutation-specific analysis (which permutations are hardest?)
- Report: per-task accuracy heatmaps

---

### E4.3: Split Fashion-MNIST

**Hypothesis Being Tested:** BIO-NN generalizes to more complex visual data in continual learning.

**Configuration:**
- Tasks: 5 binary classification tasks (T-shirt/Top vs Trouser, Pullover vs Dress, etc.)
- Training: 10 epochs per task (more complex data)
- BIO-NN: full model with all mechanisms

**Datasets:** Split Fashion-MNIST (5 tasks)

**Metrics:**
- Same as E4.1

**Expected Results:**
- BIO-NN: AA ≥ 85%, BWT ≥ -8%
- Fine-tuning: AA ≈ 70%, BWT ≈ -25%
- EWC: AA ≈ 78%, BWT ≈ -15%
- iCaRL: AA ≈ 82%, BWT ≈ -10%

**Baselines:**
- Same as E4.1

**Ablations:**
- Task granularity: 2-class vs. 5-class vs. 10-class splits
- Training epochs: 5, 10, 20
- Network capacity: small, medium, large

**Statistical Analysis:**
- Same as E4.1
- Additional: comparison across datasets (MNIST vs. Fashion-MNIST)
- Report: per-category analysis (which fashion categories are hardest?)

---

### E4.4: BIO-NN vs. EWC vs. iCaRL vs. GEM

**Hypothesis Being Tested:** BIO-NN outperforms or matches state-of-the-art continual learning methods across multiple benchmarks.

**Configuration:**
- Methods: BIO-NN, EWC, iCaRL, GEM, LwF, SI (synaptic intelligence)
- Datasets: Split MNIST, Permuted MNIST, Split Fashion-MNIST
- Hyperparameters: tuned per method (grid search)
- Seeds: 15 per method per dataset

**Datasets:** All three (Split MNIST, Permuted MNIST, Split Fashion-MNIST)

**Metrics:**
- Average accuracy (AA)
- Backward transfer (BWT)
- Forward transfer (FWT)
- Memory usage (buffer size)
- Training time
- Inference time
- Forgetting measure (FM)

**Expected Results:**
- BIO-NN: top-2 performance on all datasets
- BIO-NN: best efficiency (accuracy per compute)
- BIO-NN: lowest forgetting across all methods
- Competitive training time (within 2x of fastest method)

**Baselines:**
- All methods listed above
- Fine-tuning (lower bound)
- Joint training (upper bound)

**Ablations:**
- Buffer size sensitivity: 200, 500, 1000, 2000
- Training budget: 1, 5, 10, 20 epochs per task
- Task sequence: ascending, descending, random, grouped

**Statistical Analysis:**
- Two-way ANOVA: method × dataset
- Post-hoc: Tukey's HSD
- Critical difference diagrams
- Rank-based analysis: Friedman test
- Report: comprehensive comparison table

---

### E4.5: Continual Learning Ablation

**Hypothesis Being Tested:** Each BIO-NN mechanism contributes to continual learning performance, with structural plasticity providing the largest contribution.

**Configuration:**
- 8 conditions (mechanism on/off):
  1. No mechanisms (baseline SNN)
  2. Synaptic plasticity only
  3. Structural plasticity only
  4. Homeostatic plasticity only
  5. Synaptic + Structural
  6. Synaptic + Homeostatic
  7. Structural + Homeostatic
  8. All mechanisms (full BIO-NN)
- Dataset: Split MNIST (5 tasks)
- Seeds: 10 per condition

**Datasets:** Split MNIST, Permuted MNIST

**Metrics:**
- Average accuracy (AA)
- Backward transfer (BWT)
- Forgetting measure (FM)
- Per-task accuracy time courses
- Representation similarity (RSA)
- Network connectivity evolution

**Expected Results:**
- No mechanisms: AA ≈ 80%, BWT ≈ -30%
- Synaptic only: AA ≈ 88%, BWT ≈ -15%
- Structural only: AA ≈ 85%, BWT ≈ -18%
- Homeostatic only: AA ≈ 82%, BWT ≈ -25%
- Synaptic + Structural: AA ≈ 93%, BWT ≈ -8%
- Full BIO-NN: AA ≈ 95%, BWT ≈ -5%

**Baselines:**
- Fine-tuning (no mechanisms)
- EWC (for comparison)
- Joint training (upper bound)

**Ablations:**
- Each mechanism on/off (8 conditions)
- Mechanism interaction: 2-way and 3-way
- Task order sensitivity
- Network size sensitivity

**Statistical Analysis:**
- Factorial ANOVA: mechanism × condition
- Main effects and interactions (Type III SS)
- Effect size: η² for each mechanism
- Post-hoc: Tukey's HSD
- Report: mechanism contribution table

---

## Phase 5: Full Integration (Weeks 9-10)

**Goal:** Test full BIO-NN and conduct comprehensive ablation and efficiency analysis.

### E5.1: Full BIO-NN on Split MNIST

**Hypothesis Being Tested:** Full BIO-NN achieves state-of-the-art continual learning performance on Split MNIST.

**Configuration:**
- Architecture: 784 → 256 → 128 → 10
- All mechanisms: STDP, Hebbian, homeostatic, structural plasticity
- Hyperparameters: optimized from previous phases
- Seeds: 20 (for high-confidence results)
- Extended training: 20 epochs per task

**Datasets:** Split MNIST (5 tasks)

**Metrics:**
- All metrics from E4.1
- Additional:
  - Representation analysis (RSA, t-SNE)
  - Connectivity analysis (graph metrics)
  - Spike pattern analysis
  - Neural population dynamics

**Expected Results:**
- AA ≥ 96% (state-of-the-art)
- BWT ≥ -3% (minimal forgetting)
- FWT ≥ 5% (positive transfer)
- Better efficiency than all baselines

**Baselines:**
- All methods from E4.4
- Joint training (upper bound)
- BIO-NN variants from E4.5

**Ablations:**
- Full model vs. best ablation from E4.5
- Hyperparameter sensitivity (critical parameters)
- Architecture sensitivity (layer sizes)
- Training protocol sensitivity (epochs, learning rate)

**Statistical Analysis:**
- Confidence intervals for all metrics (95% CI)
- Comparison against all baselines (pairwise)
- Effect sizes for all comparisons
- Report: comprehensive results table

---

### E5.2: Full BIO-NN Ablation Study

**Hypothesis Being Tested:** The full BIO-NN architecture is necessary for optimal performance; removing any mechanism degrades performance.

**Configuration:**
- 8 conditions (from E4.5)
- 3 datasets: Split MNIST, Permuted MNIST, Split Fashion-MNIST
- Seeds: 15 per condition per dataset
- Full hyperparameter tuning per condition

**Datasets:** All three

**Metrics:**
- Average accuracy (AA)
- Backward transfer (BWT)
- Efficiency metrics (spike count, computation)
- Robustness metrics (noise, corruption)

**Expected Results:**
- Full BIO-NN: best performance across all datasets
- Each mechanism removal: statistically significant degradation
- Synergy: full model > sum of individual mechanisms

**Baselines:**
- All individual mechanism conditions
- Joint training (upper bound)
- Fine-tuning (lower bound)

**Ablations:**
- Full factorial: 8 conditions × 3 datasets × 15 seeds
- Hyperparameter search per condition
- Architecture search per condition

**Statistical Analysis:**
- Three-way ANOVA: mechanism × dataset × condition
- Interaction analysis: is the best mechanism combination consistent across datasets?
- Effect size: η² for each effect
- Report: comprehensive ablation table

---

### E5.3: Full BIO-NN on Permuted MNIST

**Hypothesis Being Tested:** BIO-NN generalizes to permuted MNIST, demonstrating robustness to distribution shifts.

**Configuration:**
- Same as E5.1 but on Permuted MNIST
- 10 tasks with random permutations
- Seeds: 20

**Datasets:** Permuted MNIST (10 tasks)

**Metrics:**
- Same as E5.1

**Expected Results:**
- AA ≥ 90%
- BWT ≥ -8%
- Competitive with state-of-the-art

**Baselines:**
- All methods from E4.4
- Joint training (upper bound)

**Ablations:**
- Number of tasks: 5, 10, 20
- Permutation type: random, structured, adversarial

**Statistical Analysis:**
- Same as E5.1
- Additional: task similarity analysis (do similar permutations help or hurt?)

---

### E5.4: Efficiency Analysis

**Hypothesis Being Tested:** BIO-NN is computationally more efficient than baselines while maintaining accuracy.

**Configuration:**
- Methods: BIO-NN, dense SNN, standard MLP
- Metrics: measured on identical hardware (A100 GPU)
- Inference: batch size 1 (latency) and 64 (throughput)
- Training: total FLOPS and wall-clock time

**Datasets:** MNIST (for inference), Split MNIST (for training)

**Metrics:**
- Inference latency (ms per sample)
- Throughput (samples per second)
- Training time (total hours)
- FLOPS equivalent
- Spike count (for SNNs)
- Memory usage (GB)
- Energy consumption (estimated from GPU power)

**Expected Results:**
- BIO-NN inference: 30-50% fewer spikes than dense SNN
- BIO-NN training: within 2x of standard SNN (overhead from plasticity)
- BIO-NN memory: within 20% of dense SNN
- Energy: 20-40% reduction (estimated)

**Baselines:**
- Dense SNN (100% connectivity)
- Standard MLP (no spiking)
- Structured pruning methods
- Other sparse SNN methods

**Ablations:**
- Batch size: 1, 8, 32, 64, 128
- Network size: small, medium, large
- Timestep count: 10, 20, 50, 100
- Hardware: GPU vs. CPU (and neuromorphic if available)

**Statistical Analysis:**
- Efficiency metrics comparison: paired t-tests
- Pareto frontier: accuracy vs. compute
- Cost-benefit analysis: accuracy gain per unit cost
- Report: efficiency table with all metrics

---

### E5.5: Robustness Analysis

**Hypothesis Being Tested:** BIO-NN is more robust to noise and corruption than baselines.

**Configuration:**
- Methods: BIO-NN, dense SNN, standard MLP, SNN without plasticity
- Corruptions:
  - Gaussian noise: σ = 0.05, 0.1, 0.2, 0.5
  - Salt-and-pepper noise: 5%, 10%, 20%
  - Occlusion: 10%, 20%, 30% of pixels
  - Rotation: ±10°, ±20°, ±30°
  - Scaling: 80%, 90%, 110%, 120%
- Trained on clean data, tested on corrupted data

**Datasets:** MNIST (trained), corrupted MNIST (tested)

**Metrics:**
- Accuracy under each corruption type and level
- Robustness curve: accuracy vs. corruption level
- Area under robustness curve (AUC)
- Corruption error (CE): drop from clean accuracy
- Mean corruption error (mCE): average across corruptions

**Expected Results:**
- BIO-NN: shallower accuracy degradation curve
- BIO-NN: ≥ 10% higher accuracy at highest corruption level
- BIO-NN: ≥ 15% lower mCE than baselines
- Structural plasticity contributes most to robustness

**Baselines:**
- Dense SNN (no plasticity)
- Standard MLP
- MLP with dropout (0.5)
- MLP with data augmentation (Gaussian noise injection)

**Ablations:**
- Each corruption type separately
- Mechanism contribution: which BIO-NN mechanism helps most?
- Training with noise: does training on corrupted data help more?

**Statistical Analysis:**
- Two-way ANOVA: method × corruption level
- Robustness curve comparison: bootstrap CIs
- mCE comparison: paired t-tests
- Report: robustness heatmaps and curves

---

## Phase 6: Analysis (Weeks 11-12)

**Goal:** Comprehensive analysis, visualization, and report writing.

### E6.1: Comprehensive Visualization

**Hypothesis Being Tested:** BIO-NN's biological mechanisms produce interpretable and structured representations.

**Configuration:**
- Trained BIO-NN models from E5.1-E5.3
- Visualization techniques:
  - t-SNE/UMAP of layer activations
  - Connectivity matrices (heatmaps)
  - Spike raster plots
  - Weight distribution histograms
  - Learning curves (accuracy, loss, forgetting)
  - Representational similarity matrices
  - Network graph visualization

**Datasets:** All trained models

**Metrics:**
- Cluster separability (silhouette score)
- Representation overlap between tasks
- Connectivity sparsity patterns
- Spike timing statistics
- Visual quality (subjective assessment)

**Expected Results:**
- Clear cluster separation in t-SNE plots
- Task-specific connectivity patterns
- Structured spike timing (not random)
- Interpretable weight patterns

**Baselines:**
- Visualization of baseline models (MLP, SNN, EWC, iCaRL)
- Comparison of representation quality across methods

**Ablations:**
- Visualization at different training stages
- Visualization of different mechanisms
- Visualization across different tasks

**Statistical Analysis:**
- Quantitative metrics for visualization quality
- Comparison across methods
- Report: figure gallery with annotations

---

### E6.2: Failure Analysis

**Hypothesis Being Tested:** Understanding failure modes of BIO-NN reveals paths for improvement.

**Configuration:**
- All BIO-NN models from E5.1-E5.3
- Analysis of failure cases:
  - Misclassified samples
  - High-forgetting tasks
  - Low-accuracy tasks
  - High-variance seeds
  - Efficiency failures

**Datasets:** All datasets

**Metrics:**
- Failure rate per category
- Common failure patterns
- Error analysis: confusion matrices
- Forgetting analysis: which tasks are forgotten most?
- Efficiency failures: where does BIO-NN waste computation?

**Expected Results:**
- Common failure patterns identified
- Task similarity affects forgetting
- Some mechanisms cause instability
- Clear improvement paths identified

**Baselines:**
- Failure analysis of baseline methods
- Comparison of failure modes

**Ablations:**
- Failure analysis per mechanism
- Failure analysis per dataset
- Failure analysis per task order

**Statistical Analysis:**
- Chi-squared test for failure rate differences
- Logistic regression: what predicts failure?
- Report: failure taxonomy with examples

---

### E6.3: Interpretability Analysis

**Hypothesis Being Tested:** BIO-NN's representations are more interpretable than baselines.

**Configuration:**
- Trained models from E5.1-E5.3
- Interpretability methods:
  - Neuron selectivity analysis
  - Probing classifiers (linear probes on each layer)
  - Feature visualization (maximally activating inputs)
  - Concept alignment (alignment with known concepts)
  - Representation stability analysis

**Datasets:** All datasets

**Metrics:**
- Neuron selectivity (percentage of selective neurons)
- Probe accuracy (linear classification at each layer)
- Feature quality (visual inspection)
- Concept alignment (cosine similarity)
- Stability score (representation similarity over time)

**Expected Results:**
- BIO-NN: higher neuron selectivity than baselines
- BIO-NN: higher probe accuracy in early/middle layers
- BIO-NN: more interpretable features (less adversarial-like)
- BIO-NN: more stable representations

**Baselines:**
- MLP, SNN, EWC, iCaRL
- Comparison of interpretability metrics

**Ablations:**
- Interpretability per layer
- Interpretability per mechanism
- Interpretability per task

**Statistical Analysis:**
- Comparison of interpretability metrics across methods
- Correlation: interpretability vs. accuracy
- Report: interpretability report with visualizations

---

### E6.4: Research Report Writing

**Hypothesis Being Tested:** N/A (documentation phase)

**Configuration:**
- Write comprehensive research report covering:
  - Introduction and motivation
  - Related work
  - BIO-NN architecture and mechanisms
  - Experimental results
  - Analysis and discussion
  - Conclusions and future work

**Deliverables:**
- Main paper (target: NeurIPS, ICML, or ICLR submission)
- Supplementary material (code, additional results)
- README with reproduction instructions
- Pre-trained models

**Metrics:**
- Paper completeness (all sections)
- Code quality (tests, documentation)
- Reproducibility (can results be reproduced?)

**Expected Results:**
- Submitable paper draft
- Working code repository
- All results reproducible

**Baselines:**
- Other research papers in the field
- Code quality standards

**Ablations:**
- N/A

**Statistical Analysis:**
- N/A

---

## Summary Tables

### Experiment Timeline

| Week | Phase | Experiments | Deliverables |
|------|-------|-------------|--------------|
| 1 | Foundation | E1.1, E1.2 | LIF validation, baseline SNN |
| 2 | Foundation | E1.3, E1.4 | Coding comparison, learning rules |
| 3 | Plasticity | E2.1, E2.2, E2.3 | Hebbian, STDP, R-STDP |
| 4 | Plasticity | E2.4, E2.5 | Homeostatic, ablation |
| 5 | Structure | E3.1, E3.2, E3.3 | Sparse, pruning, growth |
| 6 | Structure | E3.4, E3.5 | Combined, ablation |
| 7 | Continual | E4.1, E4.2, E4.3 | Split MNIST, permuted, Fashion |
| 8 | Continual | E4.4, E4.5 | Comparison, ablation |
| 9 | Integration | E5.1, E5.2, E5.3 | Full BIO-NN, ablation, permuted |
| 10 | Integration | E5.4, E5.5 | Efficiency, robustness |
| 11 | Analysis | E6.1, E6.2, E6.3 | Visualization, failures, interpretability |
| 12 | Analysis | E6.4 | Research report |

### Compute Budget

| Phase | GPU-Hours (est.) | Notes |
|-------|------------------|-------|
| Phase 1 | 20 | Small experiments |
| Phase 2 | 30 | Plasticity experiments |
| Phase 3 | 30 | Structural experiments |
| Phase 4 | 60 | Continual learning (many baselines) |
| Phase 5 | 50 | Full integration |
| Phase 6 | 10 | Analysis and visualization |
| **Total** | **200** | |

### Key Results to Report

| Metric | Target | Source Experiment |
|--------|--------|-------------------|
| Split MNIST AA | ≥ 96% | E5.1 |
| Split MNIST BWT | ≥ -3% | E5.1 |
| Permuted MNIST AA | ≥ 90% | E5.3 |
| Spike reduction | ≥ 30% | E5.4 |
- Robustness improvement | ≥ 10% at high noise | E5.5 |

---

## Risk Mitigation

| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| BIO-NN underperforms baselines | Extensive hyperparameter tuning; test on simple tasks first | Report negative results; analyze failure modes |
| Computational overhead too high | Profile each mechanism; optimize critical paths | Implement simplified versions; report trade-offs |
| Results don't generalize | Use 3+ diverse datasets; test on real-world data | Focus on MNIST family; note limitations |
| Statistical power insufficient | Power analysis; increase seeds if needed | Use non-parametric tests; report effect sizes |
| Reproducibility issues | Fix seeds; log everything; version control | Provide Docker containers; document dependencies |

---

## Code Organization

```
BIO-NN/
├── src/
│   ├── neurons/          # LIF, adaptive LIF, etc.
│   ├── plasticity/       # STDP, Hebbian, homeostatic
│   ├── structure/        # Pruning, growth
│   ├── networks/         # BIO-NN architecture
│   ├── training/         # Surrogate gradient, etc.
│   └── utils/            # Visualization, metrics
├── experiments/
│   ├── phase1/           # Foundation experiments
│   ├── phase2/           # Plasticity experiments
│   ├── phase3/           # Structure experiments
│   ├── phase4/           # Continual learning experiments
│   ├── phase5/           # Integration experiments
│   └── phase6/           # Analysis
├── configs/              # Experiment configurations
├── results/              # Saved results
├── figures/              # Generated figures
└── reports/              # Written reports
```

Each experiment will have:
- Configuration YAML file
- Python script to run experiment
- Analysis script
- Results directory
- Report template
