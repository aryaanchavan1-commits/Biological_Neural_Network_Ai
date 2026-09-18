# Research Gaps: What BIO-NN Aims to Address

## Gap 1: No Unified Framework Combining All Three Plasticity Mechanisms

**Current state**: SNN frameworks implement one or two of: synaptic plasticity (STDP), structural plasticity (pruning/growth), or neuromodulation (reward-modulated STDP). None combine all three in a single configurable system.

**Why it matters**: In biological brains, these mechanisms interact continuously. Synaptic weights change via STDP, synapses are created/pruned based on activity, and neuromodulators gate when and how plasticity occurs. Modeling them in isolation produces an incomplete picture.

**BIO-NN approach**: Provide all three mechanisms with configurable interactions and independent ablation hooks. Users can enable/disable any combination to study their individual and joint effects.

---

## Gap 2: Continual Learning with Jointly Adaptive Structure and Synapses

**Current state**: Continual learning research in SNNs relies on replay buffers (experience replay), weight consolidation (EWC-like regularization), or architectural isolation (progressive networks). Structural plasticity is rarely considered.

**Why it matters**: Biological continual learning likely relies on structural adaptation—forming new connections for new tasks while pruning unused ones. Current artificial approaches require external memory stores that are not biologically plausible.

**BIO-NN approach**: Enable structural plasticity (synaptogenesis, synaptic pruning) to work alongside synaptic plasticity (STDP) for continual learning, potentially eliminating the need for replay buffers.

---

## Gap 3: No Framework for Systematic Ablation of Biological Mechanisms

**Current state**: Papers compare their approach to baselines but rarely ablate individual biological mechanisms within their own framework. Results across papers are difficult to compare because they use different neuron models, learning rules, datasets, and evaluation protocols.

**Why it matters**: Without controlled ablation, it is impossible to determine which biological mechanism contributes what to performance. Claims about "bio-inspired improvement" are unsubstantiated without mechanistic isolation.

**BIO-NN approach**: Every biological mechanism (STDP, structural plasticity, neuromodulation, dendritic computation, predictive coding) is implemented as an independent, toggleable module with a standardized interface.

---

## Gap 4: Single-Mechanism Focus in Bio-Inspired Research

**Current state**: Most papers focus on one bio-inspired mechanism. STDP papers do not consider structural plasticity. Structural plasticity papers do not consider neuromodulation. Neuromodulation papers do not consider dendritic computation.

**Why it matters**: Biological systems derive their capabilities from the interaction of multiple mechanisms. Single-mechanism studies may miss emergent effects that arise only when mechanisms operate together.

**BIO-NN approach**: Implement multiple mechanisms and study their interactions, including cases where combining two mechanisms produces results that neither achieves alone.

---

## Gap 5: Lack of Unified Evaluation Across Biological Mechanisms

**Current state**: Each paper uses its own evaluation protocol, metrics, and baselines. STDP papers evaluate on spiking MNIST. Structural plasticity papers evaluate on compression metrics. Continual learning papers evaluate on task sequences.

**Why it matters**: Without unified evaluation, it is impossible to compare the relative contributions of different biological mechanisms or determine which combination is most effective for a given task.

**BIO-NN approach**: Standardized evaluation suite covering accuracy, energy efficiency (spike counts, metabolic cost), continual learning performance (forgetting rate, forward transfer), structural sparsity, and biological plausibility metrics.

---

## Gap 6: Limited Comparison of Metabolic Cost Reduction and Accuracy

**Current state**: SNN papers often claim energy efficiency but measure it indirectly (e.g., spike counts or theoretical FLOP reduction). Direct comparisons between metabolic cost reduction and accuracy trade-offs are rare.

**Why it matters**: In biological systems, metabolic cost is a first-class constraint that drives neural circuit design. Understanding the accuracy-cost trade-off in artificial SNNs is essential for neuromorphic hardware deployment.

**BIO-NN approach**: Explicitly track and report metabolic cost (spike counts, active synapses, neuron activation rate) alongside accuracy, enabling direct analysis of the accuracy-efficiency frontier.

---

## Gap 7: No Live Visualization of Internal Dynamics

**Current state**: SNN frameworks typically provide static plots of spike rasters, weight distributions, or performance curves after training. Real-time visualization of structural plasticity (synapse creation/pruning), neuromodulator levels, and dendritic activation during training is absent.

**Why it matters**: Understanding how bio-inspired mechanisms behave during learning requires observing their dynamics in real time. Static post-hoc analysis misses transient phenomena and interaction patterns.

**BIO-NN approach**: Built-in live visualization module that shows: spike raster plots, synaptic weight distributions, structural connectivity graphs, neuromodulator levels, and dendritic activation patterns during training.

---

## Gap 8: Limited Work on Interpretability of Bio-Inspired Mechanisms

**Current state**: Bio-inspired mechanisms are often treated as black boxes that improve (or don't improve) performance. Little work examines what these mechanisms actually learn or how they reorganize network structure.

**Why it matters**: Interpretability of biological mechanisms in artificial networks could inform both neuroscience (testing hypotheses about brain computation) and AI (building more interpretable systems).

**BIO-NN approach**: Provide analysis tools that track and visualize: which synapses were strengthened/weakened by STDP, which connections were created/pruned by structural plasticity, which neurons are modulated by neuromodulatory signals, and how these changes correlate with task performance.

---

## Summary Table

| # | Gap | Status Quo | BIO-NN Contribution |
|---|-----|-----------|---------------------|
| 1 | Unified plasticity framework | Mechanisms in isolation | All three plasticity types configurable |
| 2 | Continual learning + structural adaptation | Replay/consolidation only | Structural plasticity for replay-free CL |
| 3 | Systematic ablation | Ad-hoc comparisons | Toggleable modules with standardized interface |
| 4 | Multi-mechanism studies | Single-mechanism focus | Joint study of mechanism interactions |
| 5 | Unified evaluation | Fragmented benchmarks | Standardized multi-metric evaluation |
| 6 | Metabolic cost vs. accuracy | Indirect efficiency claims | Direct accuracy-cost trade-off analysis |
| 7 | Live visualization | Static post-hoc plots | Real-time dynamics monitoring |
| 8 | Interpretability | Black-box mechanisms | Tracked mechanism contributions and dynamics |
