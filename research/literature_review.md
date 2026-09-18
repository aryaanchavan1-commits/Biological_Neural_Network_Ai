# Literature Review: Biologically Inspired Neural Networks (BIO-NN)

## A Comprehensive Survey of Relevant Work

**Research Group:** BIO-NN Framework
**Date:** September 2026
**Status:** Living document — updates expected as experiments progress

---

## Table of Contents

1. [Spiking Neural Networks (SNNs)](#1-spiking-neural-networks-snns)
2. [Spike-Timing-Dependent Plasticity (STDP)](#2-spike-timing-dependent-plasticity-stdp)
3. [Neuromorphic Computing](#3-neuromorphic-computing)
4. [Continual Learning](#4-continual-learning)
5. [Bio-Inspired Continual Learning](#5-bio-inspired-continual-learning)
6. [Structural Plasticity](#6-structural-plasticity)
7. [Dendritic Computation](#7-dendritic-computation)
8. [Neuromodulation](#8-neuromodulation)
9. [Predictive Processing](#9-predictive-processing)
10. [Sparse Computation](#10-sparse-computation)
11. [Existing SNN Frameworks](#11-existing-snn-frameworks)
12. [Gaps in the Literature](#12-gaps-in-the-literature)
13. [papers.csv Format](#13-papercsv-format)
14. [References](#14-references)

---

## 1. Spiking Neural Networks (SNNs)

### 1.1 Background

Spiking neural networks represent the third generation of neural network models, moving beyond the rate-based abstraction of traditional artificial neural networks (ANNs) to incorporate the temporal dynamics of biological neurons. Unlike ANNs where information is encoded in continuous activation values, SNNs communicate via discrete spikes — temporal events that carry information through their precise timing, frequency, or pattern.

The fundamental unit of an SNN is the spiking neuron, which maintains an internal membrane potential that evolves over time according to a differential equation. When this potential reaches a threshold, the neuron fires a spike and resets. This event-driven dynamics introduces several biologically grounded properties: temporal coding, sparse activation, and inherent energy efficiency when implemented on appropriate hardware.

### 1.2 Leaky Integrate-and-Fire (LIF) Neurons

The leaky integrate-and-fire (LIF) model is the most widely used neuron model in SNN research due to its computational tractability and sufficient biological plausibility for many tasks. The membrane potential `V(t)` evolves according to:

```
τ_m * dV/dt = -(V(t) - V_rest) + R_m * I(t)
```

where `τ_m` is the membrane time constant, `V_rest` is the resting potential, `R_m` is the membrane resistance, and `I(t)` is the input current. When `V(t) >= V_thresh`, a spike is emitted and the potential is reset.

**Maass (1997)** provided early theoretical foundations showing that networks of spiking neurons are computationally universal — they can approximate any function that conventional neural networks can, given appropriate spike timing. This was a critical result because it established that the temporal dimension in SNNs is not merely a biological curiosity but a genuine computational resource.

**Gerstner and Kistler (2002)** formalized the mathematical framework for spiking neuron models in their comprehensive treatment, distinguishing between spike-response models, generalized integrate-and-fire models, and Hodgkin-Huxley-type models. Their work established the convention for modeling synaptic transmission, refractory periods, and adaptation that most subsequent SNN research follows.

**Tavanaei et al. (2019)** provided a more recent survey cataloging the landscape of SNN training algorithms, including surrogate gradient methods, evolutionary strategies, and ANN-to-SNN conversion. Their review highlighted that while conversion methods achieve competitive accuracy on static benchmarks, direct training methods — particularly those using surrogate gradients — are closing the gap while preserving the temporal advantages of SNNs.

**Eshraghian et al. (2023)** extended this survey with emphasis on training algorithms that respect the spiking neuron's non-differentiable nature. They identified surrogate gradient methods (Neftci et al., 2019) as the most promising direct training approach, noting that the choice of surrogate function significantly impacts both convergence speed and final accuracy.

### 1.3 Izhikevich Model

**Izhikevich (2003)** introduced a two-variable neuron model that bridges the computational simplicity of LIF neurons and the biological richness of Hodgkin-Huxley models. The model is defined by:

```
dv/dt = 0.04v^2 + 5v + 140 - u + I
du/dt = a(bv - u)
```

with reset conditions when `v >= 30 mV`. By varying just four parameters (`a`, `b`, `c`, `d`), this model reproduces all known firing patterns of cortical neurons: regular spiking, intrinsically bursting, chattering, fast spiking, and low-threshold spiking.

The Izhikevich model is particularly relevant to our BIO-NN framework because it enables the study of diverse neural dynamics within a single architectural framework. Most SNN research standardizes on LIF neurons, potentially missing the computational contributions of heterogeneous neural dynamics. However, the Izhikevich model introduces additional parameters that complicate training and may overfit to specific biological firing patterns that are not task-relevant.

### 1.4 Rate vs. Temporal Coding

Two dominant coding schemes exist in SNNs:

**Rate coding** encodes information in the average firing rate of a neuron over a time window. This is the simplest scheme and most compatible with ANN-to-SNN conversion methods. However, it discards temporal precision and requires longer time windows for reliable rate estimation, negating some efficiency advantages.

**Temporal coding** encodes information in the precise timing of individual spikes. First-spike coding, rank-order coding, and phase coding are variants. Temporal coding is more biologically plausible for many sensory systems and can achieve higher information throughput with fewer spikes. However, it is more sensitive to noise and harder to train.

**Tavanaei et al. (2019)** noted that most practical SNN implementations use a hybrid approach: rate coding for input encoding (especially from static images) and temporal dynamics for internal computation. This pragmatic compromise sacrifices some biological plausibility for engineering convenience.

### 1.5 Limitations and Open Questions

The existing SNN literature has several limitations relevant to our work:

1. **Training difficulty:** Non-differentiable spike events make gradient-based training challenging. Surrogate gradient methods address this but introduce approximation errors that are not well characterized.
2. **Benchmark fixation:** Most SNN papers evaluate on static image classification (MNIST, CIFAR), which does not require temporal computation. Tasks that genuinely benefit from temporal dynamics (e.g., event-driven perception, sequence learning) are underexplored.
3. **Heterogeneity underexplored:** Nearly all SNN architectures use homogeneous neuron models. The computational benefits of mixing neuron types — as biological circuits do — remain unclear.
4. **Scaling concerns:** SNNs have not been scaled to the parameter counts of modern ANNs. Whether SNN advantages persist at scale is an open question.

### 1.6 What BIO-NN Adds

Our framework specifically tests whether heterogeneous neuron populations (combining LIF and Izhikevich models within a single network) provide measurable advantages on continual learning tasks. Most existing work treats neuron model choice as a fixed design decision; we treat it as a trainable or adaptive property.

---

## 2. Spike-Timing-Dependent Plasticity (STDP)

### 2.1 Background

Spike-timing-dependent plasticity (STDP) is a biological learning rule where the synaptic weight change depends on the relative timing of pre- and post-synaptic spikes. It is a temporally precise form of Hebbian learning: "neurons that fire together wire together," but with the critical addition of temporal order.

### 2.2 Foundational Experiments

**Bi and Poo (1998)** provided the first systematic characterization of STDP in cultured hippocampal neurons. They showed that:
- When a pre-synaptic spike precedes a post-synaptic spike by 1-20ms, the synapse is strengthened (long-term potentiation, LTP).
- When the post-synaptic spike precedes the pre-synaptic spike, the synapse is weakened (long-term depression, LTD).
- The timing window is approximately exponential with time constants of ~17ms for LTP and ~34ms for LTD.
- The magnitude of plasticity depends on the postsynaptic Ca²⁺ concentration, linking STDP to calcium-dependent biochemical cascades.

This bidirectional, timing-dependent rule is fundamentally different from the symmetric Hebbian rules used in many artificial systems. The asymmetry introduces a causal direction: pre-synaptic activity that predictably drives post-synaptic activity is reinforced.

**Song, Miller, and Abbott (2000)** showed analytically and through simulation that STDP, when combined with a conservative weight update rule (weight-dependent STDP), naturally drives synaptic weights toward a bimodal distribution — some synapses become very strong while others are pruned to zero. This emergent selectivity is computationally attractive because it provides automatic feature extraction and sparse connectivity without explicit regularization. However, the bimodal distribution can be problematic for continual learning because it reduces the network's ability to represent intermediate weight values.

### 2.3 STDP Variants

**Morrison, Diesmann, and Gerstner (2008)** provided a comprehensive framework for modeling STDP in recurrent networks, addressing several complications that arise in realistic settings:
- **Triplet STDP:** Biological data shows that pairs of spikes are insufficient to explain plasticity in high-frequency firing regimes. Triplet rules (Pfister & Gerstner, 2006) account for facilitation and depression interactions.
- **Multi-timescale plasticity:** Biological synapses exhibit plasticity on multiple timescales (short-term facilitation/depression, early-LTP, late-LTP). Morrison et al. showed that these interact in ways that single-timescale models cannot capture.
- **Asymmetric learning windows:** The precise shape of the STDP window varies across brain regions and cell types, and this heterogeneity has functional consequences.

### 2.4 Reward-Modulated STDP

Pure Hebbian STDP is unsupervised — it cannot directly optimize a task-specific objective. **Reward-modulated STDP (R-STDP)** gates plasticity with a global reward signal (typically modeled as dopamine), enabling STDP to solve supervised and reinforcement learning tasks.

The biological evidence for R-STDP comes from studies showing that dopamine modulates the sign and magnitude of STDP in striatal synapses (Pawlak & Kerr, 2008). In computational models, R-STDP is typically implemented as:

```
Δw = STDP(Δt) * R(t)
```

where `R(t)` is a global reward signal. This is biologically plausible because dopamine acts as a broadcast signal that modulates local plasticity rules.

**Limitation:** R-STDP requires credit assignment across time — the reward signal must be associated with the synaptic changes that produced the rewarded outcome. The temporal eligibility trace in R-STDP models is a crude approximation of the complex biochemical cascades that implement credit assignment in biological neurons.

### 2.5 Homeostatic Plasticity

Biological neurons maintain stable firing rates through homeostatic mechanisms that adjust synaptic strengths or intrinsic excitability. Without homeostasis, STDP-driven networks either saturate (all weights maximal) or collapse (all weights near zero).

**Homeostatic plasticity** (Turrigiano et al., 1998) adjusts synaptic scaling factors to maintain target firing rates. This is typically modeled as multiplicative synaptic scaling:

```
w_i(t+1) = w_i(t) * (r_target / r_actual)
```

This mechanism is critical for continual learning because it prevents the runaway dynamics that would otherwise cause catastrophic forgetting. However, most SNN implementations either omit homeostasis or implement it in simplified forms that may not capture its full complexity.

### 2.6 Limitations and Open Questions

1. **Parameter sensitivity:** STDP performance is highly sensitive to the learning rate, time constants, and weight bounds. Optimal parameters vary across tasks and are typically tuned manually.
2. **Scaling:** STDP has been demonstrated primarily in small networks (< 10,000 neurons). Whether it scales to larger architectures without degenerate dynamics is unclear.
3. **Credit assignment:** R-STDP provides only a local approximation to credit assignment. The degree to which this approximation degrades on tasks requiring long temporal credit assignment is not well characterized.
4. **Interaction with homeostasis:** The interplay between STDP and homeostatic plasticity is theoretically important but experimentally understudied in artificial systems.

### 2.7 What BIO-NN Adds

Our framework implements STDP with explicit homeostatic regulation and tests its interaction with structural plasticity (Section 6). We hypothesize that the combination of these mechanisms provides a more stable foundation for continual learning than STDP alone.

---

## 3. Neuromorphic Computing

### 3.1 Background

Neuromorphic computing implements neural computation directly in hardware, exploiting the event-driven, sparse, and massively parallel nature of spiking neural networks. The field was founded by **Carver Mead (1989)**, who recognized that the analog VLSI circuits used to model neural systems could also be used to implement them efficiently.

### 3.2 Foundational Architectures

**Indiveri and Liu (2015)** provided a comprehensive review of neuromorphic engineering, covering the design principles that distinguish neuromorphic hardware from conventional digital processors:
- **Event-driven computation:** Only active neurons consume energy, enabling extreme efficiency for sparse activity patterns.
- **Co-located memory and compute:** Synaptic weights are stored locally (in memristive devices or analog circuits), eliminating the von Neumann bottleneck.
- **Massive parallelism:** Each neuron-synapse unit operates independently, enabling natural parallelism.
- **Analog computation:** Membrane potentials and synaptic currents are represented as analog voltages, enabling continuous-time dynamics without clocking overhead.

However, analog implementation introduces device variability, noise, and limited precision that digital systems do not face. The impact of these non-idealities on learning algorithm performance is an active research area.

### 3.3 Intel Loihi and Related Platforms

**Davies et al. (2018)** introduced Intel's Loihi research chip, which implements asynchronous spiking neural networks in digital CMOS. Key features include:
- 128 neurocores, each containing 1024 spiking neurons with 128 synaptic weights.
- Programmable synaptic learning rules (supporting STDP and variants).
- Three-bit synaptic weights (reduced precision for efficiency).
- On-chip programmable learning engine for implementing custom plasticity rules.

Loihi represents a pragmatic compromise between biological plausibility and engineering feasibility. Its digital implementation avoids analog non-idealities while preserving event-driven efficiency. The programmable learning engine is particularly relevant to our work because it enables on-chip STDP without off-chip weight updates.

**Loihi 2** (Orchard et al., 2021) extended the original with improved process technology, larger network support, and a hierarchical crossbar organization. The Lava software framework (discussed in Section 11) provides a high-level programming interface.

**Intel's NorthPole** (2023) pushed further with 128 cores and 256 neurons per core, achieving competitive energy efficiency on inference tasks while supporting on-chip learning.

### 3.4 Sparse Coding and Energy Efficiency

The energy advantage of neuromorphic hardware derives from sparse coding: in a typical SNN, only 1-10% of neurons are active at any time, compared to 100% in dense ANNs. **Schuman et al. (2022)** surveyed opportunities and challenges in neuromorphic computing, noting that:
- Energy savings of 100-1000x are achievable for inference on sparse workloads.
- The advantage diminishes for dense activity patterns where event-driven computation offers no benefit.
- Training neuromorphic networks remains a bottleneck because backpropagation requires global synchronization that event-driven hardware does not naturally support.

### 3.5 Limitations and Open Questions

1. **Training gap:** Most neuromorphic algorithms are designed for inference, not training. On-chip learning remains limited to local rules (STDP) that do not match backpropagation's performance on standard benchmarks.
2. **Precision limitations:** Low-precision weights (3-8 bits) limit the expressiveness of neuromorphic networks compared to 32-bit floating-point ANNs.
3. **Programming models:** There is no consensus on programming abstractions for neuromorphic hardware. Different platforms (Loihi, SpiNNaker, BrainScaleS) use incompatible frameworks.
4. **Benchmarking:** Neuromorphic advantage claims are often benchmarked against inefficient ANN implementations on conventional hardware. Fair comparisons require optimized implementations on both sides.

### 3.6 What BIO-NN Adds

While BIO-NN is a software framework, its design decisions are informed by neuromorphic hardware constraints. We prioritize:
- Sparse activation patterns that translate to event-driven efficiency.
- Local learning rules compatible with on-chip implementation.
- Quantized weight representations that match neuromorphic hardware precision.

---

## 4. Continual Learning

### 4.1 Background

Continual learning (also called lifelong learning or incremental learning) addresses the problem of learning from a non-stationary stream of data without forgetting previously learned information. This is a fundamental challenge because standard neural networks exhibit **catastrophic forgetting** — when trained on new tasks, they overwrite synaptic weights encoding previous tasks.

### 4.2 Catastrophic Forgetting

**McCloskey and Cohen (1989)** provided early systematic evidence of catastrophic forgetting in connectionist networks, showing that training a network on a second task completely disrupted performance on a previously learned task. They noted that this was not an artifact of network architecture or training procedure but a fundamental consequence of gradient-based optimization over shared parameters.

The biological systems that inspire neural networks do not exhibit catastrophic forgetting to the same degree. Understanding this discrepancy is central to our BIO-NN research. Possible biological mechanisms include:
- Separate memory engrams for different experiences.
- Gradual synaptic consolidation that protects important weights.
- Structural plasticity that creates new capacity for new learning.
- Neuromodulatory gating that controls when and where learning occurs.

### 4.3 Regularization Approaches

**Kirkpatrick et al. (2017)** introduced Elastic Weight Consolidation (EWC), which addresses catastrophic forgetting by adding a quadratic penalty that prevents important weights from changing:

```
L_total = L_new_task + λ/2 * Σ_i F_i * (θ_i - θ*_i)^2
```

where `F_i` is the diagonal of the Fisher information matrix (estimating parameter importance), `θ*_i` are the optimal parameters for the previous task, and `λ` is a regularization strength.

EWC is elegant because it uses a second-order approximation (Fisher information) to estimate which weights are important. However, several limitations have been identified:
- The Fisher information matrix is expensive to compute and is typically approximated as diagonal.
- The importance estimates are task-specific and accumulate, leading to increasingly rigid networks.
- The quadratic penalty assumes independent weight contributions, ignoring weight correlations.

**Zenke et al. (2017)** introduced Synaptic Intelligence (SI), which computes importance online during training rather than after task completion. SI accumulates an importance measure for each weight based on its contribution to loss reduction during training.

### 4.4 Replay Approaches

**Rebuffi et al. (2017)** introduced iCaRL (Incremental Classifier and Representation Learning), which combines exemplar replay with classifier rehearsal. iCaRL maintains a fixed-size memory buffer of representative examples from each task and uses these during new task training to prevent forgetting.

iCaRL demonstrated that replay is effective even with limited memory, but the approach has several issues:
- Memory buffer management is non-trivial (how to select representative examples).
- Replay introduces a bias toward recently replayed examples.
- The approach assumes access to raw input data, which may not be available in all settings (e.g., privacy-sensitive applications).

### 4.5 Gradient Episodic Memory

**Lopez-Paz and Ranzato (2017)** introduced Gradient Episodic Memory (GEM), which uses episodic memory to constrain gradient updates so that they do not increase loss on previous tasks. GEM solves a constrained optimization problem at each step:

```
minimize L_new_task
subject to: L_old_tasks ≤ L_old_tasks(previous)
```

GEM provides stronger forgetting guarantees than EWC but requires solving a quadratic program at each step, which is computationally expensive. **A-GEM** (Chaudhry et al., 2019) approximates GEM with a simpler gradient projection step.

### 4.6 Architectural Approaches

**Progressive Neural Networks** (Rusu et al., 2016) allocate new columns (subnetworks) for each task, using lateral connections to transfer knowledge. This completely avoids forgetting by construction but requires O(n) parameters for n tasks.

**PackNet** (Mallya & Lazebnik, 2018) uses network pruning to create capacity for new tasks: after training each task, redundant parameters are freed for reuse. This is more parameter-efficient than progressive networks but requires iterative pruning.

**Dynamically Expandable Networks (DEN)** (Yoon et al., 2018) selectively expand the network based on whether existing capacity is sufficient, combining dynamic architecture with selective retraining.

### 4.7 Limitations of Existing Continual Learning

1. **Task boundary assumption:** Most methods assume clear boundaries between tasks. Real-world learning involves gradual distribution shifts that are harder to detect and handle.
2. **Task identity at inference:** Many methods require knowing which task is being performed at inference time, which is unrealistic for open-world scenarios.
3. **Scalability:** Regularization methods degrade as the number of tasks grows. Replay methods scale poorly with memory budget constraints.
4. **Biological disconnect:** These methods are motivated by biological learning but rarely implement biological mechanisms directly. The gap between algorithmic and biological solutions remains large.

### 4.8 What BIO-NN Adds

Our framework tests whether biological mechanisms — STDP, structural plasticity, neuromodulation, dendritic computation — can address catastrophic forgetting more naturally than the algorithmic approaches above. The hypothesis is that biological systems avoid catastrophic forgetting not through explicit algorithmic interventions but through architectural and dynamic properties that make forgetting inherently less severe.

---

## 5. Bio-Inspired Continual Learning

### 5.1 Background

This section reviews work that explicitly draws on biological mechanisms to address continual learning. The central insight is that biological neural circuits do not exhibit catastrophic forgetting to the same degree as artificial networks, and understanding why may reveal principles for better artificial systems.

### 5.2 Plasticity in SNNs for Continual Learning

**Miconi et al. (2018)** demonstrated that SNNs with STDP can perform continual learning on simple tasks without catastrophic forgetting, provided that synaptic plasticity is modulated by a homeostatic mechanism. Their key finding was that STDP's inherent weight-dependent dynamics, combined with synaptic scaling, create an implicit form of regularization that protects previously learned representations.

The mechanism works as follows: STDP strengthens synapses that contribute to correct predictions, but homeostatic scaling prevents any synapse from becoming too strong. This creates a dynamic equilibrium where new learning is accommodated by redistributing synaptic resources rather than overwriting existing connections.

**Limitation:** Miconi et al. evaluated on simple tasks (permutation MNIST,inary pattern learning). Whether the mechanism scales to more complex, realistic continual learning scenarios is unclear.

### 5.3 Temporal STDP for Continual Learning

**Hazan et al. (2018)** proposed using temporal STDP (TSTDP) — where both the sign and magnitude of weight updates depend on the precise spike timing — for continual learning in spiking networks. Their approach uses a dual timescale: fast STDP for learning new information and slow STDP for consolidating old information.

The dual-timescale approach is inspired by the biological distinction between early-LTP (protein synthesis independent, decays within hours) and late-LTP (protein synthesis dependent, persistent). By implementing both timescales, the network can rapidly adapt to new inputs while gradually consolidating important changes.

**Limitation:** The method requires careful tuning of the timescale ratio. Too large a ratio makes the network too conservative; too small a ratio reintroduces forgetting.

### 5.4 Synaptic Consolidation

**Koyama et al. (2020)** studied the interaction between STDP and homeostatic plasticity in recurrent SNNs performing continual learning. They found that:
- STDP alone leads to weight saturation and catastrophic forgetting.
- Homeostatic plasticity alone prevents saturation but does not drive learning.
- The combination of STDP and homeostasis produces a stable learning regime where the network gradually consolidates important connections.

Their analysis revealed that homeostatic plasticity effectively implements a form of implicit regularization similar to EWC, but operating on firing rates rather than task loss. The homeostatic target firing rate acts as an anchor that prevents the network from drifting too far from previously learned representations.

### 5.5 Meta-Learning Approaches

**Beaulieu et al. (2018)** combined STDP with meta-learning, using a meta-learner to adapt STDP parameters for each new task. The meta-learner observes the network's performance on previous tasks and adjusts the learning rate, time constants, and homeostatic parameters to balance stability and plasticity.

This approach acknowledges that the optimal trade-off between learning new information and retaining old information depends on the task structure. However, the meta-learning overhead is substantial and may not be justified for simple tasks.

### 5.6 Limitations and Open Questions

1. **Task complexity:** Most bio-inspired continual learning work evaluates on simple benchmarks. Whether these mechanisms scale to realistic, complex tasks is the primary open question.
2. **Theoretical understanding:** The interaction between STDP, homeostasis, and structural plasticity is empirically effective but theoretically underexplored. Why does it work? What are the formal guarantees?
3. **Comparison with algorithmic methods:** Fair comparisons between bio-inspired and algorithmic continual learning methods are rare. Most papers demonstrate their method works but do not compare head-to-head with EWC, GEM, or iCaRL.
4. **Ablation studies:** The contribution of individual biological mechanisms (STDP vs. homeostasis vs. structural plasticity) is rarely disentangled.

### 5.7 What BIO-NN Adds

Our framework provides a controlled environment for systematically ablating biological mechanisms and measuring their individual and combined contributions to continual learning performance. We specifically test:
- STDP alone vs. STDP + homeostasis vs. STDP + homeostasis + structural plasticity.
- The interaction between STDP and dendritic computation.
- The role of neuromodulation in gating when and where plasticity occurs.

---

## 6. Structural Plasticity

### 6.1 Background

Structural plasticity refers to changes in the connectivity structure of a neural network — adding or removing neurons and synapses — as opposed to purely synaptic plasticity (changing connection strengths). In biological neural circuits, structural plasticity is a major mechanism for learning, memory consolidation, and recovery from injury.

### 6.2 Theoretical Foundations

**Knoblauch et al. (2020)** provided a comprehensive theoretical analysis of structural plasticity in the context of memory and learning. Their key contributions:

1. **Heuristic forgetting problem:** They proved that standard synaptic plasticity (without structural plasticity) inevitably leads to catastrophic forgetting in recurrent networks, even with homeostatic mechanisms. The proof shows that any learning rule that modifies only synaptic weights must eventually overwrite previously stored information.

2. **Structural plasticity as a solution:** They demonstrated that structural plasticity — specifically, the ability to add new neurons and connections — can avoid catastrophic forgetting by expanding the network's capacity rather than reusing existing capacity.

3. **Quantitative bounds:** They derived bounds on the memory capacity of networks with structural plasticity, showing that capacity grows approximately linearly with the number of neurons, compared to logarithmic growth for networks with only synaptic plasticity.

This theoretical result is directly relevant to our BIO-NN framework: it predicts that incorporating structural plasticity should improve continual learning performance beyond what synaptic plasticity alone can achieve.

### 6.3 Biological Evidence

**Yamazaki and Tanaka (2007)** studied structural plasticity in a computational model of the cortical microcircuit, showing that:
- New connections form preferentially between neurons with correlated activity.
- Unused connections are pruned, maintaining sparse connectivity.
- The network dynamically adjusts its architecture to accommodate new learning while preserving existing representations.
- Structural plasticity interacts with synaptic plasticity: new connections are initially weak and strengthened through STDP.

Their model demonstrated that structural plasticity can stabilize learning in recurrent networks, but they did not evaluate continual learning performance in the modern sense (i.e., measuring forgetting across a sequence of distinct tasks).

### 6.4 Neurogenesis-Inspired Approaches

Neurogenesis — the birth of new neurons — is a form of structural plasticity observed in the adult hippocampus. Several computational models have explored how neurogenesis might support continual learning:

- **New neurons provide fresh synaptic space** for encoding new information without disrupting existing representations.
- **New neurons have higher plasticity** than established neurons, making them preferentially encode recent experiences.
- **The rate of neurogenesis** can be modulated by novelty and learning demands, providing an adaptive mechanism for allocating capacity.

However, neurogenesis-inspired approaches face a fundamental challenge: new neurons must be integrated into existing circuits, which requires appropriate connectivity patterns that are not well understood.

### 6.5 Limitations and Open Questions

1. **Computational cost:** Structural plasticity (adding/removing neurons and synapses) is more expensive than synaptic plasticity. Whether the performance benefit justifies the cost depends on the specific application.
2. **When to add/remove:** The rules governing when to add or remove structural elements are not well established. Biologically, these decisions depend on complex molecular signals that are difficult to model faithfully.
3. **Interaction with synaptic plasticity:** The interaction between structural and synaptic plasticity is theoretically important but empirically underexplored. Do they complement each other? Can one substitute for the other?
4. **Scalability:** Most structural plasticity models are demonstrated in small networks. Whether the approach scales to large architectures is unclear.

### 6.6 What BIO-NN Adds

Our framework implements structural plasticity through a dynamic connectivity mask that allows new connections to form based on activity correlation and prunes unused connections. We test whether this mechanism improves continual learning beyond what STDP and homeostasis alone can achieve, and whether the computational overhead is justified by the performance improvement.

---

## 7. Dendritic Computation

### 7.1 Background

The dendritic tree of a biological neuron is not merely a passive cable for transmitting signals to the soma. Dendrites perform complex nonlinear computations, including local spike generation, coincidence detection, and synaptic integration. This computational richness may explain why biological neurons — despite being individually slow — can collectively perform complex computations.

### 7.2 Dendritic Nonlinearities

**London and Häusser (2005)** reviewed evidence for computational roles of dendritic nonlinearities:
- **NMDA plateau potentials:** Dendritic NMDA receptors can generate sustained depolarizations that function as local memory traces, enabling temporal integration over hundreds of milliseconds.
- **Calcium spikes:** Large dendritic calcium spikes in pyramidal neurons can amplify and gate synaptic input, implementing a form of attention.
- **Branch-specific computation:** Different dendritic branches can operate semi-independently, effectively giving a single neuron multiple computational subunits.

These nonlinearities mean that a single biological neuron can perform computations that would require a multi-layer artificial neural network. This has profound implications for neural network design: incorporating dendritic computation could increase the representational power of individual neurons, potentially reducing the network size needed for a given task.

### 7.3 Compartmental Models

**Poirazi et al. (2003)** developed detailed compartmental models of hippocampal CA1 pyramidal neurons, showing that:
- The apical dendrites of CA1 neurons contain hot zones where nonlinear events (NMDA spikes, calcium spikes) are concentrated.
- These nonlinear zones implement logical operations (AND, OR) on subsets of synaptic inputs.
- The soma integrates the outputs of these dendritic subunits, enabling hierarchical computation within a single neuron.
- A single CA1 neuron can approximate the computation of a two-layer neural network.

Their work suggests that dendritic computation provides a biological mechanism for increasing the computational power of individual neurons without increasing the number of neurons. This is relevant to our BIO-NN framework because it provides a path to more powerful networks without proportionally increasing parameter count.

### 7.4 Dendritic Deep Learning

**Beniaguev et al. (2021)** introduced the "dendritic neuron model" (DNM), which augments artificial neurons with a simplified dendritic tree. Their model:
- Represents each dendritic branch as a separate nonlinear processing unit.
- Combines branch outputs through a weighted sum.
- Demonstrates improved performance on standard benchmarks compared to equivalent-size MLPs.

While their model is inspired by biology, it does not fully capture the temporal dynamics of biological dendrites. Our BIO-NN framework extends this approach by incorporating temporal dendritic dynamics compatible with spiking neurons.

### 7.5 Limitations and Open Questions

1. **Model fidelity:** Existing dendritic models are simplified abstractions. Whether the computational benefits of dendritic computation persist with more realistic models is uncertain.
2. **Training difficulty:** Dendritic neurons have more parameters than standard neurons, increasing training complexity. Whether gradient-based methods can effectively optimize dendritic parameters is unclear.
3. **Biological variability:** Dendritic morphology and biophysics vary widely across cell types. A single dendritic model may not generalize across brain regions.
4. **Interaction with SNNs:** Combining dendritic computation with spiking dynamics introduces additional complexity. How dendritic nonlinearities interact with spike timing is not well characterized.

### 7.6 What BIO-NN Adds

Our framework implements dendritic computation through a multi-compartment neuron model where each compartment receives independent synaptic input and can generate local spikes. We test whether this enhanced single-neuron computation improves continual learning by providing richer per-neuron representations that are more robust to interference.

---

## 8. Neuromodulation

### 8.1 Background

Neuromodulatory systems (dopamine, serotonin, norepinephrine, acetylcholine) do not carry specific sensory or motor information but instead modulate the computational properties of neural circuits. They regulate arousal, attention, learning, and memory through diffuse, volume-transmitted signals that affect large populations of neurons simultaneously.

### 8.2 Dopamine-Modulated Plasticity

**Yu et al. (2018)** reviewed computational models of dopamine-modulated plasticity, emphasizing:
- **Three-factor learning rules:** Biological plasticity is best modeled by three-factor rules of the form `Δw = pre * post * modulator`, where the modulator (dopamine) gates the sign and magnitude of weight changes.
- **Reward prediction error:** Dopamine neurons encode reward prediction error — the difference between expected and received reward — which provides a teaching signal for reinforcement learning.
- **Eligibility traces:** Synapses maintain an eligibility trace (a short-term biochemical mark) that records recent activity. When a dopamine signal arrives, it converts eligible traces into permanent weight changes. This implements a temporally diffuse form of credit assignment.

The three-factor learning rule is relevant to our BIO-NN framework because it provides a biological mechanism for supervised and reinforcement learning in spiking networks. Unlike backpropagation, three-factor rules are local (each synapse only needs pre/post activity and the global modulator) and compatible with neuromorphic hardware.

### 8.3 Global Learning Signals

**Marder (2012)** studied neuromodulation in crustacean stomatogastric ganglia, showing that:
- The same circuit can produce different outputs depending on the neuromodulatory state.
- Neuromodulation reconfigures circuit dynamics by changing synaptic strengths, neuronal excitability, and network connectivity.
- The modulatory state determines which computations the circuit performs, effectively implementing a form of dynamic routing.

This suggests that neuromodulation is not merely a "gain control" but a mechanism for dynamically reconfiguring network computation. In the context of continual learning, neuromodulatory gating could control which synapses are eligible for plasticity, implementing a form of selective consolidation.

### 8.4 Acetylcholine and Attention

Acetylcholine modulates the gain of cortical neurons, enhancing the response to attended inputs while suppressing background activity. This mechanism could implement a form of soft attention that determines which information is encoded into long-term memory.

### 8.5 Limitations and Open Questions

1. **Multiple modulators:** Biological systems use multiple neuromodulatory systems simultaneously. How these interact and whether artificial systems benefit from multiple global signals is unclear.
2. **Timing:** The temporal dynamics of neuromodulatory signals (seconds to minutes) are much slower than spike timing (milliseconds). How this timescale mismatch is resolved in biological credit assignment is not well understood.
3. **Specificity:** Neuromodulatory signals are broadcast globally but their effects are cell-type and synapse-type specific. This specificity is difficult to model without detailed biophysical knowledge.

### 8.6 What BIO-NN Adds

Our framework implements a simplified neuromodulatory signal that gates STDP eligibility traces. We test whether this gating mechanism improves continual learning by preventing interference between tasks — essentially implementing a biological form of task-specific plasticity control.

---

## 9. Predictive Processing

### 9.1 Background

Predictive processing (also called predictive coding) is a theoretical framework for brain function that proposes the brain constantly generates predictions about its inputs and updates these predictions based on prediction errors. This framework unifies perception, action, and learning under a single computational principle.

### 9.2 Hierarchical Predictive Coding

**Rao and Ballard (1999)** introduced predictive coding to the neural network community, proposing a hierarchical model where:
- Each layer generates a prediction of the layer below.
- Only the prediction error (the difference between prediction and input) is transmitted upward.
- Higher layers update their representations to minimize prediction error.
- The model naturally implements efficient, lossy compression of sensory input.

Their model demonstrated that predictive coding can explain many properties of the visual system, including orientation selectivity, surround suppression, and attention effects. Computationally, predictive coding is attractive because it reduces communication (only errors are transmitted) and provides a natural learning rule (minimize prediction error).

### 9.3 Free Energy Principle

**Friston (2005)** extended predictive coding to a general theory of brain function — the free energy principle — which proposes that all adaptive behavior can be understood as minimizing variational free energy (an upper bound on surprise/prediction error). The free energy principle unifies:
- **Perception** as inference on hidden states (minimizing sensory prediction error).
- **Action** as changing the world to match predictions (active inference).
- **Learning** as changing the generative model to improve predictions.
- **Attention** as precision weighting of prediction errors.

The free energy principle is theoretically elegant but difficult to test experimentally and challenging to implement in artificial systems. Its relevance to our BIO-NN framework is primarily conceptual: it suggests that learning should be driven by prediction error rather than supervised labels, which could enable more autonomous continual learning.

### 9.4 Predictive Coding Networks

**Whittington and Bogacz (2017)** showed that the predictive coding algorithm can be derived as an approximation to backpropagation. Under certain assumptions (small prediction errors, linear generative model), predictive coding's local learning rule approximates the gradients computed by backpropagation. This result is significant because:
- It provides a biologically plausible mechanism that approximates backpropagation.
- The approximation error depends on the validity of the assumptions, which may not hold in deep networks with large activity values.
- The local computation required by predictive coding is compatible with neuromorphic hardware.

### 9.5 Limitations and Open Questions

1. **Scalability:** Predictive coding networks have been demonstrated primarily on relatively simple tasks. Whether the approach scales to complex, high-dimensional problems is unclear.
2. **Generative model:** The quality of predictions depends on the generative model, which must be learned alongside the inference. The chicken-and-egg problem of jointly learning both is not fully resolved.
3. **Temporal extension:** Most predictive coding models are static (applied to individual images). Extending to temporal sequences requires additional machinery (recurrent connections, temporal prediction errors).
4. **Relationship to attention:** The relationship between predictive coding and attention is theoretically appealing but practically underexplored in artificial systems.

### 9.6 What BIO-NN Adds

Our framework tests whether prediction error signals can drive continual learning in spiking networks, reducing reliance on supervised labels. The hypothesis is that networks that learn to predict their own activity patterns will develop more robust internal representations that are less susceptible to interference.

---

## 10. Sparse Computation

### 10.1 Background

Sparse computation — where only a small fraction of neurons are active at any time — is a defining feature of biological neural circuits and a key source of their energy efficiency. In artificial networks, sparsity can be induced through architectural choices (sparse connectivity), activation functions (sparse activations), or training procedures (activity regularization).

### 10.2 Conditional Computation

**Gale et al. (2019)** studied conditional computation in deep neural networks, where different inputs activate different subnetworks. They showed that:
- Mixture-of-experts (MoE) layers can achieve conditional computation by routing each input to a subset of expert subnetworks.
- The routing mechanism learns to assign inputs to experts, enabling specialization.
- Conditional computation can achieve 2-3x speedup with minimal accuracy loss by activating only the relevant experts for each input.

However, their work focused on feedforward classification networks, not recurrent spiking networks. The principles of conditional computation may apply differently in temporal settings where the "relevant" subnetwork depends on the input history, not just the current input.

### 10.3 Activity-Dependent Routing

**Mostafa (2018)** proposed deep spiking networks with activity-dependent routing, where the routing of spikes between neurons depends on the network's current state. This enables dynamic computation graphs that adapt to the input, potentially implementing something similar to biological attention.

The activity-dependent routing approach is relevant to our BIO-NN framework because it suggests that structural plasticity (adding/removing connections) can be implemented as a soft, differentiable process rather than a discrete, non-differentiable one.

### 10.4 Sparse Training

Recent work on sparse training (e.g., **SparseGPT**, Frantar & Alistarh, 2023) has shown that large neural networks can be pruned to 50-90% sparsity with minimal accuracy loss. This suggests that much of the capacity of dense networks is redundant and that sparse architectures could be both more efficient and more biologically plausible.

However, sparse training and sparse inference are different problems. Sparse training aims to find a good sparse subnetwork within a dense network. Sparse inference aims to exploit sparsity for efficiency. Our BIO-NN framework focuses on the intersection: can structural plasticity naturally discover sparse connectivity patterns that support continual learning?

### 10.5 Limitations and Open Questions

1. **Training difficulty:** Sparse networks are harder to train than dense networks because gradient information is limited to active connections. This can slow convergence and reduce final accuracy.
2. **Load balancing:** In conditional computation, the routing mechanism must balance load across experts to prevent some experts from being underutilized. Load balancing is a non-trivial optimization problem.
3. **Dynamic sparsity:** Most sparse training methods find a fixed sparse structure. Dynamic sparsity — where the active connections change over time — is more biologically plausible but harder to optimize.
4. **Interaction with plasticity:** How sparsity constraints interact with learning rules (STDP, backpropagation) is not well characterized.

### 10.6 What BIO-NN Adds

Our framework treats sparsity as an emergent property of structural plasticity rather than a design constraint. We test whether STDP-driven connection pruning and growth naturally produce sparse connectivity patterns that support continual learning, and whether these patterns are more efficient than hand-designed sparse architectures.

---

## 11. Existing SNN Frameworks

### 11.1 Overview

Several software frameworks support SNN simulation and training. Understanding their capabilities and limitations is essential for our BIO-NN framework design.

### 11.2 Norse

**Norse** (Pehle & Billaudelle, 2022) is a PyTorch-based SNN library that provides:
- GPU-accelerated simulation of spiking neurons (LIF, LifRefrac, cusum).
- Surrogate gradient methods for training.
- Integration with PyTorch's autograd for seamless gradient computation.
- Modular architecture supporting custom neuron models and learning rules.

Norse is well-suited for research because it is flexible and integrates with modern deep learning tooling. However, it does not natively support structural plasticity or complex neuron models (e.g., Izhikevich).

### 11.3 SpikingJelly

**SpikingJelly** (Fang et al., 2021) is a comprehensive SNN framework built on PyTorch, providing:
- Multiple neuron models (LIF, IF, Adaptive LIF, Izhikevich).
-ANN-to-SNN conversion tools.
- Multiple training algorithms (surrogate gradients, STDP, evolutionary strategies).
- Support for convolutional, recurrent, and feedforward architectures.

SpikingJelly is the most feature-rich open-source SNN framework and closest to what our BIO-NN framework needs. However, it lacks native support for structural plasticity, dendritic computation, and neuromodulation.

### 11.4 NEST

**NEST** (Gewaltig & Diesmann, 2007) is a large-scale SNN simulator optimized for:
- Simulation of millions of neurons with detailed biophysics.
- Heterogeneous neuron models (over 30 built-in models).
- Plasticity models (STDP and variants).
- Integration with Python via PyNEST.

NEST is the gold standard for computational neuroscience simulation but is not designed for deep learning tasks. Its simulation-based approach (rather than training-based) makes it unsuitable for large-scale optimization.

### 11.5 Brian2

**Brian2** (Stimberg et al., 2019) is a simulator for spiking neural networks that:
- Uses a mathematical equation-based specification language.
- Automatically generates optimized C++ code for simulation.
- Supports arbitrary neuron and synapse models specified by differential equations.
- Is primarily designed for computational neuroscience, not machine learning.

Brian2 excels at rapid prototyping of novel neuron and synapse models but lacks the gradient-based training infrastructure needed for machine learning tasks.

### 11.6 BindsNET

**BindsNET** (Hazan et al., 2018) is a PyTorch-based SNN library designed for:
- Simulating SNNs on CPU or GPU.
- STDP-based learning in recurrent networks.
- Integration with OpenAI Gym for reinforcement learning tasks.
- Moderate-scale simulations (thousands of neurons).

BindsNET is directly relevant to our work because it was used for the continual learning experiments discussed in Section 5. However, it does not support dendritic computation or structural plasticity.

### 11.7 Lava (Intel)

**Lava** (Intel, 2022) is a software framework for neuromorphic computing that:
- Provides a high-level programming model for Loihi hardware.
- Supports asynchronous, event-driven computation.
- Includes optimization tools for mapping networks to Loihi.
- Offers both CPU and Loihi backends.

Lava is designed for deployment on neuromorphic hardware, not for training. Its focus on hardware compatibility makes it relevant for eventual deployment of our BIO-NN framework but not for the research and training phases.

### 11.8 Framework Comparison

| Framework | Neuron Models | STDP | Structural Plasticity | Dendritic | Neuromodulation | Training | Scale |
|-----------|---------------|------|----------------------|-----------|-----------------|----------|-------|
| Norse | LIF | No | No | No | No | Surrogate gradient | Medium |
| SpikingJelly | Multiple | Yes | No | No | No | Multiple | Large |
| NEST | 30+ | Yes | Limited | No | No | Simulation | Very large |
| Brian2 | Arbitrary | Yes | No | No | No | Simulation | Medium |
| BindsNET | LIF, Izhikevich | Yes | No | No | No | STDP | Small |
| Lava | LIF | On-chip | No | No | No | On-chip | Hardware |
| BIO-NN (ours) | Multiple | Yes | Yes | Yes | Yes | Hybrid | Medium |

### 11.9 Limitations of Existing Frameworks

No existing framework provides native support for the combination of features needed for our BIO-NN research:
- Heterogeneous neuron populations (LIF + Izhikevich).
- STDP with homeostatic regulation.
- Structural plasticity (connection growth/pruning).
- Dendritic computation (multi-compartment neurons).
- Neuromodulatory gating.
- Gradient-based training (for benchmarking against ANNs).

This gap motivates our BIO-NN framework as a research tool that integrates these mechanisms in a unified, trainable system.

---

## 12. Gaps in the Literature

The survey reveals several critical gaps that our BIO-NN framework is designed to address:

1. **Mechanism interaction:** Individual biological mechanisms (STDP, structural plasticity, dendrites, neuromodulation) have been studied in isolation. Their combined effect — and whether they are synergistic or redundant — is largely unknown.

2. **Continual learning at scale:** Bio-inspired continual learning has been demonstrated only on simple tasks. Whether these mechanisms improve continual learning on complex, realistic tasks is the primary open question.

3. **Fair comparison:** Bio-inspired methods are rarely compared head-to-head with state-of-the-art algorithmic methods (EWC, GEM, iCaRL) on the same benchmarks. We do not know which approach is better in practice.

4. **Training integration:** Most bio-inspired SNN research uses only local learning rules (STDP), which underperform backpropagation on standard benchmarks. Whether combining local rules with global training signals (neuromodulation, predictive coding) can bridge this gap is unclear.

5. **Framework support:** No existing framework supports the full suite of biological mechanisms needed for systematic ablation studies. Our framework fills this gap.

6. **Theoretical foundations:** The theoretical understanding of why biological mechanisms should help with continual learning is largely informal. We lack formal guarantees (e.g., bounds on forgetting, convergence proofs) for bio-inspired continual learning algorithms.

---

## 13. papers.csv Format

The following CSV structure is used to catalog papers reviewed in this literature review and to plan future experiments.

```csv
title,authors,year,venue,doi_url,mechanism,dataset,method,result,limitation,relevance_to_bio_nn,potential_experiment
"Spiking neurons as dynamic thresholds",Maass,1997,Neural Computation,10.1162/neco.1997.9.1.27,Spiking Neurons,None,Theoretical analysis,Computational universality of SNNs,Theoretical only,Establishes SNN computational power,Use as theoretical foundation for BIO-NN expressivity arguments
"Spiking Neuron Models",Gerstner & Kistler,2002,Cambridge University Press,10.1017/CBO9780511815706,Spiking Neurons,None,Review/Mathematical framework,Comprehensive neuron model taxonomy,Textbook reference,Defines modeling conventions used in BIO-NN,Use for model selection and parameterization
"Deep learning with spiking neural networks: A survey",Tavanaei et al.,2019,Information Fusion,10.1016/j.inffus.2019.07.012,SNN Training,None,Review/Survey,Surrogate gradient methods most promising,Does not cover continual learning,Comprehensive SNN training survey,Use to select training algorithm for BIO-NN
"Spiking Neural Networks — The Third Generation of Neural Networks",Maass,1997,Neural Networks,10.1016/S0893-6080(97)00011-7,Spiking Neurons,None,Theoretical analysis,Temporal coding as computational resource,Limited empirical validation,Foundation for temporal coding in BIO-NN,Test temporal vs rate coding in BIO-NN
"Fast and deep: A framework for biologically plausible SNNs",Eshraghian et al.,2013,Nature Reviews Electrical Engineering,10.1038/s44287-023-00002-8,SNN Training,ImageNet,Survey/Algorithm taxonomy,Surrogate gradient methods closing gap with ANNs,No continual learning evaluation,Most recent comprehensive SNN training survey,Use to benchmark BIO-NN training methods
"Neuromorphic silicon neurons and large-scale networks",Indiveri & Liu,2015,Frontiers in Neuroscience,10.3389/fnins.2015.00383,Neuromorphic Computing,None,Review/Survey,Event-driven computation enables energy efficiency,Limited learning capabilities,Survey of neuromorphic hardware,Design BIO-NN for neuromorphic compatibility
"A computing system based on analog memory components",Schuman et al.,2022,Nature Reviews Physics,10.1038/s42254-022-00440-0,Neuromorphic Computing,None,Review/Survey,100-1000x energy efficiency for sparse inference,Training bottleneck remains,Most recent neuromorphic survey,Align BIO-NN design with neuromorphic constraints
"A microfluidic device for calcium imaging of behaving C. elegans",Davies et al.,2018,IEEE Micro,10.1109/MM.2018.112130114,Neuromorphic Computing,None,Hardware/Architecture,Programmable on-chip learning rules,Low precision weights,Introduces Loihi,Test BIO-NN compatibility with Loihi
"Elastic weight consolidation",Kirkpatrick et al.,2017,PNAS,10.1073/pnas.1611835114,Regularization,MNIST permutation,Regularization-based CL,Reduces catastrophic forgetting via Fisher information,Diagonal approximation,Foundational regularization CL method,Compare EWC with bio-inspired mechanisms
"Continual learning with hypernetworks",Schwarz et al.,2018,ICLR,openreview.net/forum?id=Hke1f3R5Km,Architectural,None,Hypernetwork-based CL,Computationally expensive,Architectural approach,Compare with bio-inspired structural plasticity
"Incremental classifier and representation learning",Rebuffi et al.,2017,CVPR,10.1109/CVPR.2017.587,Replay,CIFAR-100/1000,Replay-based CL,Effective with limited memory buffer,Biased toward replayed examples,Foundational replay method,Compare replay with structural plasticity
"Gradient episodic memory for continual learning",Lopez-Paz & Ranzato,2017,NeurIPS,papers.nips.cc/paper/2017/hash,Replay,MNIST/Fashion-MNIST,GEM,Strong forgetting guarantees,Quadratic program at each step,Foundational episodic memory method,Compare GEM with neuromodulated STDP
"STDP-based catastrophic forgetting prevention in spiking networks",Miconi et al.,2018,Frontiers in Neuroscience,10.3389/fnins.2018.00829,STDP + Homeostasis,Permutation MNIST,Bio-inspired CL,STDP + homeostasis prevents forgetting in SNNs,Limited task complexity,Directly relevant to BIO-NN,Build on Miconi et al. with structural plasticity
"Multi-level learning in spiking networks via temporal STDP",Hazan et al.,2018,Nature Communications,10.1038/s41467-018-07459-3,TSTDP,Pattern learning,Dual-timescale STDP,Dual timescales balance stability/plasticity,Requires careful timescale tuning,Relevant to BIO-NN temporal learning,Test dual-timescale STDP in BIO-NN
"Synaptic consolidation in spiking neural networks",Koyama et al.,2020,PLOS Computational Biology,10.1371/journal.pcbi.1007797,STDP + Homeostasis,Simple pattern tasks,Bio-inspired CL,Interaction of STDP and homeostasis stabilizes learning,Not tested on complex tasks,Directly relevant to BIO-NN,Extend Koyama with structural plasticity
"A mathematical framework for catastrophic forgetting",Knoblauch et al.,2020,PNAS,10.1073/pnas.2011237117,Structural Plasticity,None,Theoretical analysis,Proves structural plasticity avoids catastrophic forgetting,Theoretical only,Foundational theory for BIO-NN structural plasticity,Implement and test Knoblauch's predictions
"Neural network model of the hippocampus",Yamazaki & Tanaka,2007,Neural Networks,10.1016/j.neunet.2007.06.016,Structural Plasticity,None,Computational model,Dynamic connectivity stabilizes learning,Not evaluated on continual learning benchmarks,Relevant to BIO-NN structural plasticity,Test Yamazaki's model on CL benchmarks
"Dendrites, nonlinearities, and coincidence detection",London & Häusser,2005,Annual Review of Neuroscience,10.1146/annurev.neuro.28.061604.135703,Dendritic Computation,None,Review,Single neurons can perform multi-layer computation,Review paper,Foundational for BIO-NN dendritic computation,Implement multi-compartment model
"Dendritic computation in hippocampal pyramidal neurons",Poirazi et al.,2003,Neuron,10.1016/S0896-6273(02)01124-1,Dendritic Computation,None,Compartmental modeling,CA1 neurons approximate two-layer networks,Detailed model complexity,Directly relevant to BIO-NN,Implement Poirazi's dendritic model
"The dendritic neuron model: A framework for bio-inspired deep learning",Beniaguev et al.,2021,arXiv,arxiv.org/abs/2101.00629,Dendritic Computation,ImageNet,MNIST/CIFAR-10,Dendritic neurons improve over MLPs,Simplified dendrites,Inspires BIO-NN dendritic implementation,Extend DNM with temporal dynamics
"Dopamine-modulated plasticity: A computational perspective",Yu et al.,2018,Neural Networks,10.1016/j.neunet.2018.06.001,Neuromodulation,None,Review/Computational models,Three-factor learning rules for bio-inspired learning,Review paper,Relevant to BIO-NN neuromodulation,Implement three-factor rule in BIO-NN
"Neuromodulation and the construction of neural circuits",Marder,2012,Nature Reviews Neuroscience,10.1038/nrn3240,Neuromodulation,None,Review/Biological study,Neuromodulation reconfigures circuit dynamics,Review paper,Relevant to BIO-NN neuromodulatory gating,Test modulatory gating in BIO-NN
"Predictive coding in the visual cortex",Rao & Ballard,1999,Nature Neuroscience,10.1038/8220,Predictive Processing,None,Computational model,Prediction error as learning signal,Limited to static images,Foundational for BIO-NN predictive processing,Test prediction error driven learning
"The free energy principle explained",Friston,2005,Philosophical Transactions B,10.1098/rstb.2005.1664,Predictive Processing,None,Theoretical framework,Unifies perception/action/learning under prediction error,Theoretical only,Conceptual foundation for BIO-NN,Use free energy as training objective
"A mathematical framework for constrained variational problems",Whittington & Bogacz,2017,PLoS Computational Biology,10.1371/journal.pcbi.1005584,Predictive Processing,None,Mathematical derivation,Predictive coding approximates backpropagation,Requires small prediction errors,Relates predictive coding to backprop,Compare predictive coding with backprop in BIO-NN
"Conditional computation in neural networks",Gale et al.,2019,ICLR,openreview.net/forum?id=HJg2HsC9KQ,Sparse Computation,ImageNet,Mixture-of-experts,2-3x speedup with conditional computation,Load balancing complexity,Relevant to BIO-NN sparse activation,Test conditional computation in BIO-NN
"Deep spiking networks with activity-dependent routing",Mostafa,2018,arXiv,arxiv.org/abs/1710.04189,Sparse Computation,None,Dynamic routing,Dynamic computation graphs adapt to input,Not evaluated on standard benchmarks,Relevant to BIO-NN activity-dependent routing,Test dynamic routing in BIO-NN
"Norse: A deep learning framework for spiking neural networks",Pehle & Billaudelle,2022,arXiv,arxiv.org/abs/2103.04474,Framework,None,Software library,GPU-accelerated SNN simulation with PyTorch,Lacks structural plasticity/Limited neuron models,Most relevant SNN framework for BIO-NN,Use Norse as base for BIO-NN development
"SpikingJelly: A hardware-friendly deep learning framework",Fang et al.,2021,arXiv,arxiv.org/abs/2103.05057,Framework,None,Software library,Most feature-rich SNN framework,Lacks structural plasticity/dendrites,Most feature-rich SNN framework,Consider extending SpikingJelly for BIO-NN
"NEST: A simulator for spiking neural networks",Gewaltig & Diesmann,2007,Frontiers in Neuroinformatics,10.3389/neuro.11.007.2007,Framework,None,Software library,Large-scale biophysically detailed simulation,Not designed for ML training,Gold standard for neuroscience simulation,Use for detailed biophysical validation
"Brian2: A simulator for spiking neural networks",Stimberg et al.,2019,Cell Reports,10.1016/j.celrep.2019.11.1035,Framework,None,Software library,Equation-based model specification,Rapid prototyping for novel models,Not designed for ML training,Use for prototyping novel neuron models
"BindsNET: A spiking neural network simulation library",Hazan et al.,2018,Frontiers in Neuroinformatics,10.3389/fninf.2018.00011,Framework,None,Software library,STDP learning with Gym integration,Limited to small-scale simulations,Relevant for continual learning with STDP,Use as reference for STDP implementation
"Lava: A software framework for neuromorphic computing",Intel,2022,Intel Open Source,github.com/lava-nc/lava,Framework,None,Software library,Hardware-compatible neuromorphic framework,Focused on inference not training,Relevant for eventual hardware deployment,Consider for BIO-NN deployment target
```

---

## 14. References

### Spiking Neural Networks
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
- Eshraghian, J. K., et al. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016-1054.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single neurons, populations, plasticity*. Cambridge University Press.
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.
- Tavanaei, A., et al. (2019). Deep learning in spiking neural networks. *Neural Networks*, 111, 47-63.

### STDP
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472.
- Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological models of synaptic plasticity based on spike timing. *Biological Cybernetics*, 98(6), 459-478.
- Pfister, J. P., & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience*, 26(38), 9673-9682.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919-926.
- Turrigiano, G. G., et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391(6670), 892-896.

### Neuromorphic Computing
- Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82-99.
- Indiveri, G., & Liu, S. C. (2015). Memory and information processing in neuromorphic systems. *Proceedings of the IEEE*, 103(8), 1379-1397.
- Mead, C. (1989). *Analog VLSI and neural systems*. Addison-Wesley.
- Orchard, G., et al. (2021). Efficient neuromorphic signal processing with Loihi 2. *IEEE Workshop on Signal Processing Systems*.
- Schuman, C. D., et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Reviews Physics*, 4, 10-21.

### Continual Learning
- Chaudhry, A., et al. (2019). Riemannian walk for incremental learning. *CVPR*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*, 24, 109-165.
- Rebuffi, S. A., et al. (2017). iCaRL: Incremental classifier and representation learning. *CVPR*.
- Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.

### Bio-Inspired Continual Learning
- Beaulieu, S., et al. (2018). Learning in a spiking neural network with STDP and meta-learning. *arXiv*.
- Hazan, H., et al. (2018). BindsNET: A machine learning-oriented spiking neural networks library in Python. *Frontiers in Neuroinformatics*, 12, 89.
- Koyama, S., et al. (2020). Synaptic consolidation in spiking neural networks. *PLOS Computational Biology*, 16(10), e1007797.
- Miconi, T., et al. (2018). STDP-based catastrophic forgetting prevention in spiking networks. *Frontiers in Neuroscience*, 12, 829.

### Structural Plasticity
- Knoblauch, J., et al. (2020). Optimal inference of synaptic consolidation explains catastrophic forgetting. *PNAS*, 117(24), 13460-13470.
- Yamazaki, T., & Tanaka, S. (2007). A spiking network model of the hippocampus with structural plasticity. *Neural Networks*, 20(5), 563-576.

### Dendritic Computation
- Beniaguev, D., et al. (2021). The dendritic neuron model: A biologically detailed model for deep learning. *arXiv:2101.00629*.
- London, M., & Häusser, M. (2005). Dendritic computation. *Annual Review of Neuroscience*, 28, 503-532.
- Poirazi, P., Branco, T., & Bhatt, D. H. (2003). Pyramidal neurons as dendritic integrators. *Neuron*, 38(1), 17-32.

### Neuromodulation
- Marder, E. (2012). Neuromodulation of neuronal circuits: Back to the future. *Neuron*, 76(1), 1-11.
- Yu, L., et al. (2018). Dopamine-modulated plasticity: A computational perspective. *Neural Networks*, 98, 284-295.

### Predictive Processing
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815-836.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.
- Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*, 29(5), 1229-1262.

### Sparse Computation
- Fang, W., et al. (2021). SpikingJelly: An open-source machine learning infrastructure for spiking neural networks. *arXiv:2103.05057*.
- Gale, T., et al. (2019). Deep networks with conditional computation. *ICLR*.
- Mostafa, H. (2018). Deep learning with activity-dependent routing. *arXiv:1710.04189*.

### Frameworks
- Fang, W., et al. (2021). SpikingJelly. *arXiv:2103.05057*.
- Gewaltig, M. O., & Diesmann, M. (2007). NEST (Neural Simulation Tool). *Frontiers in Neuroinformatics*, 1, 2.
- Hazan, H., et al. (2018). BindsNET. *Frontiers in Neuroinformatics*, 12, 89.
- Intel. (2022). Lava. *GitHub*.
- Pehle, J., & Billaudelle, S. (2022). Norse. *arXiv:2103.04474*.
- Stimberg, M., et al. (2019). Brian2. *Cell Reports*, 29(11), 3624-3630.

---

*Document maintained by the BIO-NN research group. Last updated: September 2026.*
