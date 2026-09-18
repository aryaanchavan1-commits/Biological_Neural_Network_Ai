# Limitations: Honest Assessment

## 1. Biological Neurons Are Vastly More Complex Than Any Computational Model

A biological neuron contains thousands of ion channels, complex dendritic arbors with hundreds of branches, intracellular signaling cascades, gene expression dynamics, and glial cell interactions. Our model captures a fraction of this complexity.

**Impact**: Results from BIO-NN cannot be directly extrapolated to biological systems. We model useful abstractions, not biological reality.

**Mitigation**: We explicitly document which biological features are abstracted away and maintain clear separation between what is biologically grounded and what is an engineering approximation.

---

## 2. Our Models Are Abstractions, Not Accurate Representations

STDP in BIO-NN is a simplified Hebbian rule with exponential time windows. Structural plasticity follows a threshold-based rule rather than the complex molecular signaling that governs synaptogenesis in vivo. These are engineering tools inspired by biology, not attempts to replicate biology.

**Impact**: Claims about "biological plausibility" should be interpreted loosely. We are inspired by biological mechanisms, not implementing them faithfully.

**Mitigation**: We label all mechanisms with their abstraction level (high/medium/low fidelity to biology) and include references to the biological literature for each mechanism.

---

## 3. Computational Constraints Limit Model Size

Biological brains contain ~86 billion neurons with ~100 trillion synapses. Our models are limited to thousands of neurons due to computational constraints, particularly when simulating structural plasticity (dynamic graph operations) and neuromodulation (global state updates).

**Impact**: Results on small networks may not scale to larger architectures. Emergent phenomena that require massive scale will not be observed.

**Mitigation**: We provide scaling analyses where possible and note when computational constraints may affect conclusions. We optimize structural plasticity operations for GPU execution but acknowledge inherent scaling limits.

---

## 4. STDP Implementations May Not Capture Real STDP Dynamics

Real STDP depends on precise spike timing at sub-millisecond resolution, involves multiple molecular pathways (NMDA receptors, calcium dynamics), and exhibits heterosynaptic effects. Our exponential STDP window is a common simplification.

**Impact**: The specific learning dynamics observed in BIO-NN may differ from those achievable with more biologically accurate STDP models.

**Mitigation**: We support configurable STDP parameters (time windows, learning rates) and provide guidance on how to adjust them for closer approximation to experimental STDP data.

---

## 5. Structural Plasticity Rules Are Simplified

Biological structural plasticity involves axon guidance molecules, cell adhesion molecules, trophic factors, and activity-dependent gene expression. Our implementation uses local activity thresholds to determine synapse creation and pruning.

**Impact**: The structural changes observed in BIO-NN may not match the spatial and temporal patterns of biological structural plasticity.

**Mitigation**: We acknowledge this limitation explicitly and frame structural plasticity as an engineering tool inspired by biology, not a biological simulation.

---

## 6. Neuromodulation Is Approximated

Real neuromodulation involves multiple neurotransmitter systems (dopamine, serotonin, acetylcholine, norepinephrine), each with distinct effects on plasticity, and operates through complex receptor dynamics. Our implementation uses a single global modulatory signal.

**Impact**: The nuanced effects of different neuromodulatory systems (e.g., dopamine for reward, acetylcholine for attention) are not captured.

**Mitigation**: We support multiple modulatory signals with distinct effects and provide documentation on how to configure them for different neuromodulatory hypotheses.

---

## 7. Dataset Scale Is Limited

BIO-NN is evaluated on MNIST, Fashion-MNIST, and CIFAR-10. These are small datasets by modern standards. Large-scale evaluation on ImageNet, language tasks, or real-world continual learning scenarios is computationally prohibitive with our current implementation.

**Impact**: Performance claims are limited to small-scale benchmarks and may not generalize to complex, high-dimensional problems.

**Mitigation**: We focus on controlled experiments that isolate the effects of individual mechanisms rather than claiming state-of-the-art performance on large benchmarks.

---

## 8. Results on MNIST/Fashion-MNIST May Not Generalize

MNIST and Fashion-MNIST are simple, low-resolution image classification tasks. The biological mechanisms that provide advantages on these tasks may not provide advantages on more complex problems, or may provide different kinds of advantages.

**Impact**: Conclusions about the utility of bio-inspired mechanisms are task-dependent and should not be generalized without further experimentation.

**Mitigation**: We include CIFAR-10 as a more complex benchmark and note when results are specific to simple tasks. We encourage replication on additional benchmarks.

---

## 9. Surrogate Gradients Are Not Biologically Plausible

To enable gradient-based training of SNNs, we use surrogate gradients (e.g., sigmoid or piecewise linear approximations to the spike function derivative). This is a training-time convenience that has no biological counterpart.

**Impact**: The training process is not biologically plausible. Only the inference-time dynamics (spiking, structural changes, neuromodulation) have biological grounding.

**Mitigation**: We clearly distinguish between training mechanisms (surrogate gradients, backpropagation) and inference mechanisms (STDP, structural plasticity, neuromodulation). The framework supports STDP-based training without surrogate gradients for experiments requiring biological plausibility.

---

## 10. Energy Efficiency Claims Require Hardware Validation

We report metabolic cost proxies (spike counts, active synapse counts) but do not have access to neuromorphic hardware for direct energy measurements. Actual energy efficiency depends on hardware implementation details that are not captured by our simulations.

**Impact**: Claims about energy efficiency are theoretical and may not translate to actual energy savings on neuromorphic hardware.

**Mitigation**: We provide raw metrics (spike counts, synaptic operations) that can be mapped to energy estimates on specific neuromorphic platforms. We do not claim absolute energy efficiency numbers.

---

## 11. We Cannot Claim Biological Realism

BIO-NN is inspired by biology, not a model of biology. The purpose is to explore whether bio-inspired mechanisms can improve artificial neural network capabilities, not to make claims about how the brain works.

**Impact**: Neuroscience conclusions drawn from BIO-NN experiments should be treated as hypotheses, not evidence. The framework is a tool for AI, not a neuroscience simulator.

**Mitigation**: We maintain clear documentation distinguishing engineering objectives from neuroscience hypotheses. We cite neuroscience literature for biological grounding but do not claim our models validate or refute biological theories.

---

## 12. Continual Learning Benchmarks May Not Reflect Real-World Continual Learning

Standard continual learning benchmarks (split MNIST, permuted MNIST, CIFAR-100 sequences) use artificially constructed task boundaries. Real-world continual learning involves overlapping tasks, gradual distribution shifts, and no explicit task boundaries.

**Impact**: Performance on standard benchmarks may not predict performance on realistic continual learning scenarios.

**Mitigation**: We note this limitation and provide experiments on overlapping task distributions where possible. We recommend evaluating on more realistic benchmarks as they become available.

---

## Summary

These limitations are inherent to any computational model of biological systems. They do not invalidate the research but should be considered when interpreting results. The value of BIO-NN lies not in replicating biology but in exploring whether biological mechanisms can inspire new engineering approaches to continual learning, energy efficiency, and adaptive neural networks.
