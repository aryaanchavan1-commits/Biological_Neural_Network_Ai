# BIO-NN Research Hypotheses

## Primary Research Question

**"Can adaptive synaptic/structural plasticity combined with sparse spiking computation improve continual learning while maintaining competitive accuracy and reducing active computation?"**

This question drives the entire BIO-NN research program. It encompasses three interrelated dimensions: learning performance (continual learning retention), computational efficiency (sparse spiking), and biological plausibility (adaptive plasticity mechanisms). The hypotheses below decompose this question into testable, falsifiable claims with explicit experimental designs.

---

## Null Hypothesis

### H0: No Significant Improvement

**Statement:** BIO-NN with biological mechanisms (spiking neurons, synaptic plasticity, structural plasticity, homeostatic regulation) does not significantly improve continual-learning retention over standard baselines (MLP, standard SNN without plasticity, rehearsal-based methods).

**Independent Variables:**
- Network architecture type (BIO-NN vs. baseline)
- Presence/absence of biological mechanisms

**Dependent Variables:**
- Average accuracy across all tasks (final accuracy)
- Backward transfer (BWT)
- Forward transfer (FWT)
- Forgetting measure (average accuracy drop per task over time)

**Controlled Variables:**
- Total number of parameters (matched across architectures)
- Training epochs per task
- Learning rate schedules
- Batch size
- Dataset (same split MNIST, permuted MNIST, etc.)
- Hardware (same GPU for fair timing comparison)
- Random seeds (minimum 10 seeds per condition)

**Statistical Tests:**
- Two-sided Welch's t-test (unequal variance) between BIO-NN and each baseline
- If multiple baselines: one-way ANOVA followed by Tukey's HSD post-hoc
- Significance threshold: α = 0.05 with Bonferroni correction for multiple comparisons
- Non-parametric alternative: Mann-Whitney U test if normality assumption fails (Shapiro-Wilk test)

**Effect Size Thresholds:**
- Cohen's d ≥ 0.5 (medium effect) for practical significance
- Report 95% confidence intervals for all effect sizes
- Minimum detectable effect: d = 0.4 with power = 0.8, n = 10 seeds

**Failure Criteria:**
- H0 is NOT rejected if p > 0.05 after correction for all primary metrics
- H0 is NOT rejected if effect size d < 0.3 for average accuracy
- H0 is NOT rejected if BIO-NN shows higher variance (coefficient of variation > 1.5x baseline)

**Follow-up if H0 is supported (BIO-NN fails):**
1. Diagnostic analysis: are plasticity mechanisms interfering with gradient-based learning?
2. Hyperparameter sensitivity sweep to rule out poor tuning
3. Mechanism-level ablation to identify which component hurts performance
4. Compare against stronger baselines (e.g., experience replay with larger buffers)
5. Test on simpler tasks to determine if failure is task-dependent
6. Consider whether biological mechanisms require longer training horizons

**Follow-up if H0 is rejected (BIO-NN works):**
1. Proceed to H1-H3 for detailed mechanism analysis
2. Identify minimum mechanism subset needed for improvement
3. Scale experiments to larger datasets and architectures

---

## Primary Hypotheses

### H1: Improved Continual Learning Retention

**Statement:** BIO-NN improves continual-learning retention, measured by average accuracy and forgetting metrics, compared to baselines without plasticity and structural mechanisms.

**Rationale:** Biological neural circuits naturally resist catastrophic forgetting through synaptic consolidation, structural remodeling, and homeostatic regulation. BIO-NN's incorporation of these mechanisms should enable more stable retention of previously learned tasks while acquiring new ones.

**Independent Variables:**
- BIO-NN with full biological mechanisms (primary condition)
- Baseline SNN without plasticity (control 1)
- Standard MLP with SGD (control 2)
- EWC (Elastic Weight Consolidation) baseline (control 3)
- iCaRL baseline (control 4)
- GEM baseline (control 5)

**Dependent Variables:**
- **Average Accuracy (AA):** Mean accuracy across all tasks after final task training
  - Formula: AA = (1/T) Σ acc(T,i) where T is total tasks, acc(T,i) is accuracy on task i after training task T
- **Backward Transfer (BWT):** Measure of forgetting
  - Formula: BWT = (1/(T-1)) Σ [acc(T,i) - acc(i,i)] for i = 1 to T-1
  - Negative BWT indicates forgetting; closer to 0 is better
- **Forward Transfer (FWT):** Measure of positive transfer
  - Formula: FWT = (1/(T-1)) Σ [acc(i-1,i) - acc(init,i)] where acc(init,i) is accuracy on task i before training it
- **Forgetting Measure:** Maximum accuracy drop per task
  - Formula: FM = (1/(T-1)) Σ max_j∈{1,...,i-1} [acc(j,i) - acc(T,i)]
- **Learning Curve Area Under Curve (AUC):** Integrated performance across training
- **Performance Stability:** Variance of accuracy across tasks (lower is better)

**Controlled Variables:**
- Network capacity: total parameter count matched across all architectures
  - BIO-NN: adjust hidden layer sizes to match baseline parameter count
  - Report both matched-parameter and unconstrained comparisons
- Training protocol: same number of epochs per task (e.g., 5 epochs for MNIST tasks)
- Data ordering: identical task sequences across all methods
- Buffer size for replay methods: fixed at 200, 500, 1000 samples
- Optimization: Adam optimizer with lr=1e-3 for all methods unless otherwise noted
- Architecture depth: 2-3 hidden layers for all methods

**Statistical Tests:**
- Primary: One-way ANOVA across all methods (if assumptions met)
- Post-hoc: Tukey's HSD pairwise comparisons
- Non-parametric: Kruskal-Wallis test followed by Dunn's test
- Repeated measures: If using multiple seeds, mixed-effects ANOVA with seed as random factor
- Correction: Bonferroni or Holm-Bonferroni for family-wise error
- Minimum 15 seeds per condition for adequate power

**Effect Size Thresholds:**
- Cohen's d ≥ 0.6 for BIO-NN vs. best baseline (practical significance)
- η² ≥ 0.06 for ANOVA (medium effect)
- Report effect sizes with 95% CI for all comparisons
- Minimum: BIO-NN must beat best non-biological baseline by ≥ 2% accuracy with p < 0.05

**Failure Criteria:**
- BIO-NN does not achieve statistically significant improvement over best baseline
- BIO-NN achieves significance but effect size d < 0.3 (negligible practical benefit)
- BIO-NN shows high variance across seeds (CV > 2x baseline CV)
- Forgetting metric is worse than rehearsal-based baselines
- Improvement only on one dataset but not others (lack of generalization)

**Follow-up if H1 is supported:**
1. Identify which biological mechanism contributes most (test via ablation in E5.2)
2. Scale to more tasks (10, 20, 50 task sequences)
3. Test on more complex datasets (CIFAR-100, miniImageNet)
4. Investigate neural correlates: which neurons/connections are most stable?
5. Test with different task ordering (easy-first, hard-first, random)
6. Investigate whether performance degrades with very long task sequences

**Follow-up if H1 is refuted:**
1. Analyze per-task learning curves: is the issue acquisition or retention?
2. Check if plasticity mechanisms cause instability during training
3. Test with longer training per task to rule out insufficient adaptation time
4. Compare learning dynamics: do BIO-NN representations overlap more?
5. Consider whether biological mechanisms are redundant with architectural inductive biases

---

### H2: Reduced Active Computation

**Statement:** BIO-NN reduces active computation, measured by spike count, firing rate, and active synapses, while maintaining competitive performance compared to non-biological baselines.

**Rationale:** Biological neural circuits are remarkably energy-efficient, using sparse coding and event-driven computation. BIO-NN's sparse spiking should reduce the total number of operations needed for inference and training, translating to lower computational cost.

**Independent Variables:**
- BIO-NN with sparse spiking (primary condition)
- Dense SNN (all neurons fire every timestep) (control 1)
- Standard MLP (continuous activations) (control 2)
- BIO-NN with varying sparsity targets (10%, 20%, 50% of neurons active)

**Dependent Variables:**
- **Spike Count:** Total number of spikes per forward pass
  - Measured per layer and total
- **Firing Rate:** Average spike rate across all neurons
  - Formula: (total spikes) / (number of neurons × number of timesteps)
- **Active Synapses:** Percentage of synapses that transmit non-zero values per forward pass
  - Measured as: (synapses with non-zero input × non-zero output) / total synapses
- **Computational Cost:** FLOPS equivalent or spike-based operations
  - For SNN: count of synaptic events (pre-spike × weight × post-threshold)
  - For MLP: standard FLOPS count
- **Energy Estimate:** Estimated energy consumption using benchmark models
  - Use CPU/GPU power models or direct measurement on neuromorphic hardware
- **Inference Latency:** Time to classify one sample (wall-clock time)
- **Throughput:** Samples processed per second
- **Sparsity Pattern Analysis:** Distribution of active neurons across time
  - Coefficient of variation of firing rates across neurons
  - Temporal sparsity: percentage of timesteps with zero spikes

**Controlled Variables:**
- Task and dataset (same for all conditions)
- Accuracy constraint: all methods must achieve ≥ X% accuracy on held-out set
  - If BIO-NN achieves lower accuracy, report efficiency only at matched accuracy levels
- Network size: same total parameters
- Hardware: same device, same precision (fp32 or fp16)
- Inference protocol: same number of timesteps for SNNs (e.g., 10-50 timesteps)
- Batch size: 1 for fair latency comparison, 64 for throughput

**Statistical Tests:**
- Paired t-test: BIO-NN vs. each baseline on matched samples
- Regression analysis: spike count vs. accuracy trade-off curve
- Correlation: Pearson/Spearman between sparsity level and accuracy
- Efficiency ratio: (accuracy / computational_cost) compared across methods
- Minimum 1000 inference samples per condition

**Effect Size Thresholds:**
- ≥ 50% reduction in active computation (spike count or FLOPS) with ≤ 2% accuracy drop
- Cohen's d ≥ 0.8 for efficiency metrics (large effect)
- Energy reduction ≥ 30% at matched accuracy
- Statistical significance: p < 0.01 (stricter due to practical importance)

**Failure Criteria:**
- BIO-NN does not reduce computation by ≥ 30% vs. dense baseline
- Reduction in computation comes with > 5% accuracy drop (unacceptable trade-off)
- Efficiency gain only appears at very low accuracy levels
- Computational overhead of plasticity mechanisms negates sparsity gains
- Spike-based computation is slower than equivalent dense operations on available hardware

**Follow-up if H2 is supported:**
1. Profile which layers contribute most to efficiency gains
2. Test on neuromorphic hardware (Intel Loihi, IBM TrueNorth) for real energy measurements
3. Investigate adaptive sparsity: does BIO-NN learn to allocate computation where needed?
4. Measure energy during training (not just inference)
5. Test with different timestep budgets to find optimal efficiency-accuracy trade-off
6. Explore whether efficiency gains transfer to larger models

**Follow-up if H2 is refuted:**
1. Analyze spike patterns: is sparsity uniform or concentrated in uninformative neurons?
2. Check if surrogate gradients create dense backward passes despite sparse forward
3. Test with event-driven inference (only compute when spikes occur)
4. Compare against structured sparsity methods (e.g., channel pruning)
5. Consider if biological sparsity requires different hardware to realize benefits

---

### H3: Improved Robustness Under Distribution Shifts

**Statement:** BIO-NN improves robustness under distribution shifts (noise, corruption, out-of-distribution) compared to non-biological baselines.

**Rationale:** Biological neural circuits are inherently robust due to redundant coding, homeostatic regulation, and structural diversity. These properties should help BIO-NN maintain performance when inputs deviate from the training distribution.

**Independent Variables:**
- BIO-NN with full biological mechanisms (primary condition)
- Standard MLP (control 1)
- SNN without plasticity (control 2)
- Standard MLP with dropout (control 3)
- Standard MLP with data augmentation (control 4)

**Dependent Variables:**
- **Noise Robustness:** Accuracy under additive Gaussian noise
  - Noise levels: σ = 0.05, 0.1, 0.2, 0.5, 1.0
  - Plot accuracy vs. noise level curve
  - Measure: area under robustness curve (higher = more robust)
- **Corruption Robustness:** Accuracy under common corruptions
  - Use CIFAR-10-C / MNIST-C benchmark corruptions
  - 15 corruption types × 5 severity levels
  - Mean corruption error (mCE): lower is better
- **Out-of-Distribution (OOD) Detection:** AUROC for distinguishing ID vs OOD
  - OOD datasets: different but related (e.g., MNIST → notMNIST, Fashion-MNIST)
  - Metrics: AUROC, AUPR, FPR at 95% TPR
- **Adversarial Robustness:** Accuracy under PGD adversarial attacks
  - ε = 0.1, 0.2, 0.3 (L∞ norm)
  - Attack strength: 20, 50, 100 steps
- **Temporal Noise:** Robustness to temporal jitter in spike timing
  - Randomly shift spike times by ±1-5 timesteps
- **Missing Data:** Accuracy with randomly masked input features
  - Mask 10%, 20%, 30%, 50% of input dimensions

**Controlled Variables:**
- Base accuracy: all methods must have comparable clean accuracy (within 2%)
- Model capacity: same parameter count
- Training data: identical training sets
- Test data: same corrupted/shifted test sets
- Random seeds: 10+ per condition per noise level
- Preprocessing: identical normalization and preprocessing

**Statistical Tests:**
- Two-way ANOVA: method × noise level (interaction effect tests differential robustness)
- Area under curve comparison: bootstrap CIs for AUC differences
- OOD detection: DeLong test for AUROC comparison
- Robustness curve comparison: permutation test on paired accuracy differences
- Report: accuracy at each noise level with confidence intervals

**Effect Size Thresholds:**
- ≥ 10% higher accuracy at highest noise level (d ≥ 0.7)
- mCE reduction ≥ 15% across all corruption types
- AUROC improvement ≥ 5% for OOD detection
- Accuracy degradation slope: BIO-NN slope shallower than baseline by ≥ 30%

**Failure Criteria:**
- BIO-NN shows no improvement in robustness over baselines
- Robustness improvement only at extreme noise levels (not practical range)
- BIO-NN is more fragile to specific corruption types
- OOD detection is worse than baseline (BIO-NN is overconfident on OOD)
- Adversarial robustness is not improved (biological mechanisms don't help against adversarial examples)

**Follow-up if H3 is supported:**
1. Identify which mechanism drives robustness (ablation study)
2. Test on real-world distribution shifts (e.g., different lighting, camera angles)
3. Investigate theoretical basis: why do biological mechanisms help robustness?
4. Test on time-series data where temporal robustness is critical
5. Combine with adversarial training for stronger robustness
6. Measure calibration: are BIO-NN confidence estimates better aligned?

**Follow-up if H3 is refuted:**
1. Analyze failure modes: which shifts are most problematic?
2. Test if robustness requires specific mechanism combinations
3. Consider whether robustness needs explicit training (not just architecture)
4. Compare against dedicated robustness methods (e.g., Gaussian augmentation, TRADES)
5. Investigate whether spike-based computation inherently loses information needed for robustness

---

## Secondary Hypotheses

### H4: Mechanism Contribution Analysis

**Statement:** Among BIO-NN's biological mechanisms, synaptic plasticity (STDP/Hebbian), structural plasticity (growth/pruning), and homeostatic regulation contribute differentially to continual learning performance, with structural plasticity providing the largest contribution to forgetting reduction.

**Rationale:** While all biological mechanisms theoretically contribute to stable learning, structural plasticity (adding/removing connections) may be most impactful because it allows the network to physically reorganize to accommodate new tasks without overwriting existing representations.

**Independent Variables:**
- Full BIO-NN (all mechanisms)
- BIO-NN - structural plasticity (synaptic + homeostatic only)
- BIO-NN - synaptic plasticity (structural + homeostatic only)
- BIO-NN - homeostatic plasticity (synaptic + structural only)
- BIO-NN - synaptic only
- BIO-NN - structural only
- BIO-NN - homeostatic only
- BIO-NN - no biological mechanisms (baseline)

**Dependent Variables:**
- Same as H1 (AA, BWT, FWT, FM)
- Per-task forgetting breakdown
- Representation similarity analysis (RSA) between tasks

**Statistical Tests:**
- Factorial ANOVA: mechanism type × presence/absence
- Contribution analysis: variance explained by each mechanism (Type III SS)
- Relative importance analysis: Lindeman-Merenda-Gold method
- 8 conditions × 15 seeds = 120 runs minimum

**Effect Size Thresholds:**
- Each mechanism must explain ≥ 5% of variance in BWT to be considered meaningful
- Structural plasticity must have ≥ 1.5x larger effect than synaptic plasticity for H4 to be supported
- Interaction effects must be reported (synergy between mechanisms)

**Failure Criteria:**
- No single mechanism explains > 10% of variance (mechanisms are redundant)
- Mechanisms interfere with each other (negative interaction effects)
- All mechanisms contribute equally (no differentiation)

**Follow-up:**
- If structural plasticity dominates: focus optimization on growth/pruning algorithms
- If synaptic plasticity dominates: investigate learning rule variants
- If homeostatic dominates: explore regulation mechanisms for other domains
- If synergy dominates: mechanisms must be used together (no simplification possible)

---

### H5: Metabolic Cost-Accuracy Correlation

**Statement:** BIO-NN's metabolic cost reduction (measured by total synaptic events and spike count) is negatively correlated with accuracy, but the correlation is weaker than in non-biological baselines, indicating BIO-NN achieves a better efficiency-accuracy trade-off.

**Rationale:** There is a fundamental trade-off between computation and accuracy. Biological systems navigate this trade-off efficiently through adaptive resource allocation. BIO-NN should shift the Pareto frontier of this trade-off.

**Independent Variables:**
- BIO-NN with varying sparsity targets (5%, 10%, 20%, 50%, 80% of neurons active)
- Baseline MLP with varying widths (25%, 50%, 100%, 200% of standard size)
- SNN with varying threshold levels (affecting sparsity)

**Dependent Variables:**
- **Pareto Frontier:** Accuracy vs. computational cost curve
  - X-axis: total synaptic events (or FLOPS)
  - Y-axis: test accuracy
- **Efficiency Ratio:** Accuracy per unit of computation
  - Formula: accuracy / (total spikes × weight_updates)
- **Knee Point:** The point of diminishing returns on the Pareto frontier
  - Measured by maximum curvature of the accuracy-computation curve
- **Metabolic Cost Model:** Estimated ATP equivalent for biological plausibility
  - Use standard neuroenergetic models (e.g., Attwell & Laughlin 2001)

**Statistical Tests:**
- Correlation: Pearson r and Spearman ρ between cost and accuracy
- Comparison of correlation coefficients: Steiger's Z-test for dependent correlations
- Pareto dominance: count of non-dominated solutions per method
- Curve fitting: fit power law y = ax^b and compare exponents

**Effect Size Thresholds:**
- BIO-NN correlation r should be ≤ -0.3 (moderate negative correlation expected)
- BIO-NN correlation should be significantly weaker than baseline (p < 0.05)
- BIO-NN should have ≥ 20% more Pareto-optimal points than baseline
- Knee point: BIO-NN should achieve same accuracy at 30% lower computation

**Failure Criteria:**
- BIO-NN shows same or stronger negative correlation than baselines
- BIO-NN has fewer Pareto-optimal points than baseline
- Efficiency gains only appear at very low accuracy levels
- Metabolic cost model doesn't apply to BIO-NN's computation pattern

**Follow-up:**
- Test on neuromorphic hardware for real energy measurements
- Investigate adaptive computation: does BIO-NN allocate more resources to hard examples?
- Explore dynamic resource allocation during inference
- Model biological energy budgets for constraint satisfaction

---

### H6: Improved Interpretability

**Statement:** BIO-NN's biological mechanisms improve model interpretability, measured by the alignment of learned representations with human-interpretable features and the stability of learned representations over time.

**Rationale:** Biological neural circuits develop structured, hierarchical representations that neuroscientists can interpret. BIO-NN's biological constraints may encourage similar structured representations, making the model more interpretable than unconstrained baselines.

**Independent Variables:**
- BIO-NN with all mechanisms
- Standard MLP
- SNN without plasticity
- BIO-NN with only synaptic plasticity
- BIO-NN with only structural plasticity

**Dependent Variables:**
- **Neuron Selectivity:** Percentage of neurons with high selectivity for specific classes
  - Selectivity = (max_class_firing - mean_other_firing) / max_class_firing
- **Concept Alignment:** Alignment of learned features with known concepts
  - Use probe classifiers on intermediate representations
  - Measure: linear separability of concepts in representation space
- **Representation Stability:** Cosine similarity of representations across training
  - Track how representations of old tasks change during new task learning
  - Formula: sim(t) = cosine_sim(R_task1_at_time_t, R_task1_at_time_0)
- **Neuron Utilization:** Percentage of neurons that are significantly active
  - Dead neurons: firing rate < 1% of max
  - Overactive neurons: firing rate > 90% of max
- **Visualization Quality:** t-SNE/UMAP cluster separability score
  - Silhouette score of task/class clusters
- **Sparse Coding Efficiency:** How well sparse representations capture information
  - Mutual information between sparse code and input

**Statistical Tests:**
- Comparison of selectivity distributions: KS test
- Representation stability: mixed-effects model with time as fixed effect
- Cluster quality: ANOVA on silhouette scores across methods
- Probe accuracy: comparison of linear probe performance
- Minimum 5 models per condition, 3 random seeds

**Effect Size Thresholds:**
- ≥ 20% higher neuron selectivity in BIO-NN (d ≥ 0.5)
- ≥ 15% higher representation stability (lower forgetting in representation space)
- ≥ 10% higher silhouette score for cluster separability
- ≥ 15% more neurons with meaningful selectivity (not dead or overactive)

**Failure Criteria:**
- BIO-NN shows lower interpretability than baselines
- Interpretability gains only in early layers, not deep representations
- Stability comes from not learning (low plasticity), not from robust representations
- Visualization quality is not meaningfully different

**Follow-up:**
- If interpretability improves: develop tools for BIO-NN visualization
- Test if interpretability helps with debugging and model improvement
- Explore whether BIO-NN representations are more aligned with neuroscience findings
- Investigate if interpretability transfers across tasks in continual learning

---

### H7: Structural Plasticity Superiority

**Statement:** Structural plasticity (dynamic connection growth and pruning) reduces catastrophic forgetting more than synaptic plasticity alone, because structural changes create dedicated pathways for new tasks while preserving existing pathways.

**Rationale:** Synaptic plasticity modifies the strength of existing connections, which can interfere with previously learned information. Structural plasticity adds new connections, allowing the network to expand capacity without disrupting existing representations.

**Independent Variables:**
- BIO-NN with structural plasticity only
- BIO-NN with synaptic plasticity only
- BIO-NN with both (full model)
- BIO-NN with neither (baseline SNN)
- Structural plasticity variants: growth only, pruning only, growth + pruning

**Dependent Variables:**
- Same as H1 (AA, BWT, FWT, FM)
- Connection overlap: percentage of connections shared between tasks
- Network size over time: number of active connections as tasks are learned
- Task-specific pathway analysis: degree of pathway specialization

**Statistical Tests:**
- Factorial ANOVA: plasticity type × growth/pruning
- Contrast analysis: structural vs. synaptic specific comparisons
- Pathway analysis: graph metrics (modularity, community structure)
- Longitudinal analysis: repeated measures across task presentation

**Effect Size Thresholds:**
- Structural plasticity must reduce forgetting by ≥ 25% more than synaptic plasticity alone
- Network size should increase by ≤ 50% while maintaining performance
- Pathway specialization: modularity > 0.3 in BIO-NN vs. < 0.1 in synaptic-only
- Connection overlap between consecutive tasks: < 60% in BIO-NN, > 80% in synaptic-only

**Failure Criteria:**
- Structural plasticity provides no additional benefit over synaptic plasticity
- Growth leads to unbounded network expansion without performance gain
- Pruning removes important connections, hurting retention
- Structural changes are random, not task-directed
- Computational cost of structural changes outweighs benefits

**Follow-up:**
- If structural plasticity dominates: optimize growth/pruning rules
- If both contribute: find optimal ratio of structural to synaptic change
- Test on hardware that supports dynamic connectivity (memristive devices)
- Investigate biological plausibility of structural plasticity rules

---

## Summary of Hypothesis Relationships

```
H0 (Null)
  |
  +--> H1 (Retention) ---+--> H4 (Mechanism Contribution)
  |                       |
  +--> H2 (Efficiency) ---+--> H5 (Cost-Accuracy Trade-off)
  |                       |
  +--> H3 (Robustness) ---+--> H6 (Interpretability)
                          |
                          +--> H7 (Structural > Synaptic)
```

H0 must be rejected before H1-H3 are tested. H1-H3 are tested in parallel. H4-H7 are secondary analyses that build on H1-H3 results. Each hypothesis has clear success/failure criteria and follow-up experiments.

---

## Experiment Mapping

| Hypothesis | Primary Experiments | Supporting Experiments |
|-----------|-------------------|----------------------|
| H1 | E4.1-E4.5, E5.1-E5.3 | E2.1-E2.5, E3.1-E3.5 |
| H2 | E5.4 | E1.1-E1.4, E3.1-E3.5 |
| H3 | E5.5 | E1.1-E1.4 |
| H4 | E5.2 | E2.1-E2.5, E3.1-E3.5 |
| H5 | E5.4 | E3.1-E3.5 |
| H6 | E6.3 | E4.1-E4.5 |
| H7 | E3.1-E3.5 | E5.2 |

---

## Statistical Power Analysis

**Minimum Sample Sizes:**
- Primary experiments (H1-H3): 15 seeds per condition
- Secondary experiments (H4-H7): 10 seeds per condition
- Ablation studies: 10 seeds per condition
- Efficiency measurements: 1000 samples per condition (for inference timing)

**Power Calculations:**
- Target power: 0.80
- Minimum detectable effect: d = 0.5 (medium)
- α = 0.05 (corrected for multiple comparisons)
- With 15 seeds: power = 0.81 for d = 0.5 (two-sided t-test)
- With 10 seeds: power = 0.72 for d = 0.5 (acceptable for secondary hypotheses)

**Multiple Comparisons Correction:**
- Primary hypotheses (H1-H3): Bonferroni correction (α = 0.05/3 = 0.0167)
- Secondary hypotheses (H4-H7): Holm-Bonferroni (less conservative)
- Post-hoc comparisons: Tukey's HSD (controls family-wise error)
- Report: adjusted p-values for all comparisons

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BIO-NN performs worse than baselines | Medium | High | Extensive hyperparameter tuning; test on simple tasks first |
| Mechanisms interfere with each other | Medium | Medium | Systematic ablation; test mechanisms independently first |
| Computational overhead negates efficiency gains | Medium | High | Profile each mechanism; optimize critical paths |
| Results don't generalize across datasets | Low | High | Use 3+ diverse datasets; test on real-world data |
| Statistical power insufficient | Low | Medium | Power analysis before experiments; increase seeds if needed |
| Reproducibility issues | Low | High | Fix random seeds; log all hyperparameters; version control |
