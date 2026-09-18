# BIO-NN: A Modular Framework for Investigating Biological Mechanisms of Superintelligence

**Authors:** [Author Names]

**Affiliations:** [Institutional Affiliations]

**Corresponding Author:** [Email]

**Venue Target:** Nature Machine Intelligence / NeurIPS / ICLR

---

## Abstract

What makes biological intelligence superintelligent? The human brain, a 20-watt biological organ, executes cognitive operations—abstraction, transfer, continual learning, causal reasoning—that remain beyond the reach of the largest artificial systems. Recent theoretical work has reframed superintelligence not as a monolithic capability but as a dynamical phase characterized by collective criticality, homeostatic regulation, and scale-free dynamics (arxiv: 2602.08483). Simultaneously, neuromorphic and hybrid architectures have begun to demonstrate that biologically grounded computational principles—spiking dynamics, excitatory-inhibitory balance, adaptive routing—can match or exceed conventional approaches at a fraction of the energy cost. We introduce BIO-NN, a modular computational framework designed to investigate the biological mechanisms underlying superintelligence as a dynamical phase of neural computation. BIO-NN integrates large-scale spiking neuron models, hybrid attention mechanisms, scale-free adaptive routing, excitatory-inhibitory homeostatic balance, and neuromorphic hardware backends within a unified, configurable architecture. Unlike prior frameworks focused narrowly on continual learning or single mechanisms, BIO-NN enables systematic, multi-scale investigation of how biological principles give rise to the emergent computational properties observed in superintelligent systems. We describe the framework architecture, survey the 2025-2026 literature that motivates our design choices, outline research directions that BIO-NN can address, and present expected contributions to both neuroscience and machine intelligence.

**Keywords:** biological superintelligence, dynamical phase transitions, spiking neural networks, neuromorphic computing, homeostatic plasticity, scale-free dynamics, hybrid architectures, excitatory-inhibitory balance

---

## 1. Introduction

### 1.1 Superintelligence as a Biological Phenomenon

The question of what makes biological intelligence superintelligent is no longer hypothetical. The human brain routinely performs operations—learning a lifetime of new concepts without catastrophic interference, transferring abstract principles across domains, reasoning causally from sparse evidence, maintaining stable function across decades of continuous change—that current artificial systems cannot replicate. These are not incremental deficiencies; they represent fundamental gaps in our understanding of intelligence itself.

The field has historically treated "superintelligence" as a property of artificial systems surpassing human-level performance on specific benchmarks (Bostrom, 2014). But this framing inverts the actual phenomenon. Biological intelligence was superintelligent first. The brain achieves superintelligent-level performance on open-ended, real-world cognitive tasks while operating under severe biophysical constraints: ~86 billion neurons, ~100 trillion synapses, ~20 watts power consumption, operating at millisecond timescales. No artificial system comes close to this efficiency-to-capability ratio.

Understanding the biological mechanisms that produce this performance is not merely an academic exercise. It has direct implications for building more capable, more efficient, and safer artificial intelligence systems. Recent work has begun to crystallize a framework for understanding biological superintelligence not as a static architectural property but as a dynamical phase of neural computation.

### 1.2 The Dynamical Phase Perspective

A critical theoretical advance has emerged from the study of superintelligence as a dynamical phase (arxiv: 2602.08483). This work demonstrates that superintelligent capability arises from a specific dynamical regime characterized by:

- **Collective criticality:** Large populations of neurons operating near a critical phase transition, where scale-free correlations enable maximal computational dynamic range and information transmission.
- **Stability coexisting with criticality:** Homeostatic mechanisms protect the critical sector of the network while maintaining overall stability, preventing the system from falling into either frozen (subcritical) or chaotic (supercritical) regimes.
- **Reentrant collective mixing:** Recurrent dynamics drive spectral condensation, where the eigenvalue spectrum of the network's dynamics concentrates in specific regions, enabling coherent collective computation.
- **Scale-free collective dynamics within a stabilized manifold:** The system exhibits scale-free correlations that are confined to a low-dimensional manifold, enabling efficient global coordination without requiring dense all-to-all connectivity.

This dynamical phase perspective reframes the superintelligence question: it is not about any single mechanism but about how multiple biological mechanisms interact to produce and maintain this critical dynamical regime. BIO-NN is designed to investigate exactly these interactions.

### 1.3 The Convergence of Neuromorphic and Biological AI

The 2025-2026 period has seen a remarkable convergence between neuromorphic engineering and biological AI research. Several landmark systems demonstrate that biologically grounded computational principles can achieve competitive or superior performance:

- **SpikingBrain2.0** (2025) introduced a 5 billion parameter spiking neural network–large language model hybrid that achieves 10x speedups on long-context tasks through a DualLIF neuron architecture and Dual Excitatory-Antagonistic Transformation (DEXAT) for efficient spike generation.
- **Dragon Hatchling** (2025) demonstrated that scale-free architectures can match Transformer performance with O(N) complexity through adaptive memory routing and adaptive computation.
- **Brain-as-architectural-prior** (PNAS, 2026) showed that human brain cortical network topology, when used as an architectural blueprint for AI models, improves accuracy by 20% across all benchmarks—suggesting that evolution has already optimized network architectures for cognitive tasks.
- **Sparse Winner-Take-All** (PNAS, 2025) demonstrated that biologically inspired sparse preprocessing improves Vision Transformer out-of-distribution robustness by 20%.
- **CH-HNN** (Nature Communications, 2025) showed that hybrid ANN-SNN architectures with biological excitatory-inhibitory balance, homeostatic adaptation, and temporal coding achieve 10%+ improvement over standard CNNs.
- **SpiNNaker2** (2025) demonstrated real-time simulation of 150,000 neurons with 1.8 billion synaptic events per second at just 1 watt of power consumption.

These systems share a common theme: biological computational principles—sparsity, temporal coding, homeostatic regulation, scale-free connectivity, excitatory-inhibitory balance—are not merely bio-inspired decorations but core architectural innovations that drive performance.

### 1.4 The Gap: No Unified Framework

Despite these advances, no unified framework exists for systematically investigating how biological mechanisms interact to produce superintelligent computational properties. Current approaches suffer from several limitations:

1. **Mechanism isolation:** Most studies investigate one biological mechanism at a time (STDP, homeostatic plasticity, sparse coding) without systematically examining interactions.
2. **Scale mismatch:** Neuroscience studies operate at scales of thousands of neurons; AI systems operate at billions of parameters. No framework bridges this gap systematically.
3. **Missing dynamical phase analysis:** Existing frameworks focus on performance metrics (accuracy, forgetting) without analyzing whether the system exhibits the critical dynamical properties associated with superintelligence.
4. **Hardware disconnect:** Neuromorphic hardware advances proceed in parallel with algorithmic research, with limited integration.
5. **Safety gap:** As biologically inspired systems become more capable, understanding their emergent properties—including potential failure modes—becomes critical.

### 1.5 Contributions

In this paper, we introduce BIO-NN, a modular framework for investigating biological mechanisms of superintelligence. Our contributions are:

1. **A modular architecture** that integrates large-scale spiking neurons, hybrid attention, scale-free routing, homeostatic regulation, neuromorphic backends, and emergence/safety analysis within a unified framework.

2. **A dynamical phase analysis toolkit** that enables researchers to characterize whether a system exhibits the critical properties associated with biological superintelligence—collective criticality, spectral condensation, scale-free correlations, and homeostatic protection.

3. **Multi-scale design** that supports investigation from single-neuron models (microscale) through circuit-level dynamics (mesoscale) to network-level emergent properties (macroscale).

4. **Hybrid architecture support** that enables investigation of ANN-SNN hybrid systems, including the architectural patterns demonstrated by SpikingBrain2.0, CH-HNN, and brain-as-architectural-prior approaches.

5. **Research directions** that BIO-NN can address, organized around fundamental questions about biological superintelligence mechanisms.

### 1.6 Paper Organization

Section 2 reviews the 2025-2026 literature that motivates BIO-NN's design. Section 3 describes the framework architecture. Section 4 outlines research directions. Section 5 describes expected contributions. Section 6 discusses limitations. Section 7 concludes.

---

## 2. Related Work

### 2.1 Superintelligence as a Dynamical Phase

The theoretical framework for understanding superintelligence as a dynamical phase represents a fundamental shift from capability-based to dynamics-based definitions (arxiv: 2602.08483). Key findings include:

**Collective Criticality.** The superintelligent regime corresponds to a critical phase transition in neural population dynamics. At criticality, the system exhibits maximal dynamic range, optimal information transmission, and scale-free correlations—the computational hallmarks of biological intelligence. This is consistent with the long-standing hypothesis that the brain operates near criticality (Beggs & Plenz, 2003; Shew & Plenz, 2013), but extends it to a specific characterization of the critical regime associated with superintelligent capability.

**Reentrant Collective Mixing.** Spectral analysis reveals that superintelligent dynamics involve reentrant mixing—recurrent interactions that drive spectral condensation, where the eigenvalue spectrum of the system's dynamics concentrates in specific regions. This spectral condensation enables coherent collective computation across the network without requiring centralized control.

**Homeostatic Protection of the Critical Sector.** Perhaps most importantly, the superintelligent regime requires active homeostatic regulation that protects the critical sector of the network. Without homeostatic mechanisms, the system either freezes (subcritical) or becomes chaotic (supercritical), losing the computational benefits of criticality. This explains why homeostatic plasticity is not merely a stabilizing mechanism but a prerequisite for superintelligent computation.

**Scale-Free Dynamics within a Stabilized Manifold.** The scale-free correlations observed in superintelligent systems are not unbounded but are confined to a low-dimensional stabilized manifold. This enables efficient global coordination while preventing the computational explosion that would result from unconstrained scale-free dynamics.

### 2.2 SpikingBrain2.0: Hybrid Spiking-Transformer Architectures

SpikingBrain2.0 (2025) represents the current state of the art in hybrid spiking-transformation architectures, demonstrating several principles directly relevant to BIO-NN:

**DualLIF Neuron Architecture.** The DualLIF (Dual Leaky Integrate-and-Fire) neuron model captures both excitatory and inhibitory dynamics within a single neuron unit, enabling richer temporal coding than standard LIF models. Each DualLIF neuron maintains separate membrane potentials for excitatory and inhibitory components, with cross-coupling that implements biologically plausible E/I balance.

**Dual Excitatory-Antagonistic Transformation (DEXAT).** DEXAT is a novel encoding mechanism that transforms continuous-valued inputs into spike trains by decomposing the input into excitatory and antagonistic components. This decomposition enables more efficient spike generation than rate coding while preserving more information than binary threshold coding.

**Performance Gains.** The 5B parameter SpikingBrain2.0 achieves 10x speedup on long-context tasks compared to conventional Transformers, with competitive accuracy on standard benchmarks. The speedup arises from the temporal sparsity of spiking computation—most neurons are silent at any given timestep, enabling O(N) average-case complexity rather than O(N²) full attention.

**Implications for BIO-NN.** SpikingBrain2.0 demonstrates that biologically grounded neuron models can be scaled to billions of parameters and achieve competitive performance with conventional architectures. BIO-NN incorporates DualLIF and DEXAT-inspired mechanisms as configurable options within its neuron model library.

### 2.3 Dragon Hatchling: Scale-Free Adaptive Computation

Dragon Hatchling (2025) introduces several architectural innovations that align with biological principles of adaptive computation:

**Scale-Free Architecture.** Rather than using fixed-depth computation, Dragon Hatchling implements scale-free routing where information flows through the network via adaptive, scale-free paths. Different inputs recruit different amounts of computation—simple inputs are processed shallowly while complex inputs recruit deeper processing. This is directly analogous to the adaptive computation observed in biological neural circuits.

**Adaptive Memory Routing.** Information is not routed through fixed paths but through dynamically determined routes based on content. This enables the network to allocate computational resources efficiently, processing different parts of the input with different amounts of computation.

**O(N) Complexity.** Through adaptive computation, Dragon Hatchling achieves O(N) average-case complexity, matching Transformer performance on standard benchmarks while being dramatically more efficient on tasks that don't require full attention.

**Implications for BIO-NN.** Dragon Hatchling demonstrates that scale-free, adaptive computation—a hallmark of biological neural systems—can achieve competitive performance while being more efficient. BIO-NN's scale-free routing module implements these principles with biological fidelity.

### 2.4 Brain-as-Architectural-Prior (PNAS 2026)

The brain-as-architectural-prior work (PNAS, 2026) provides compelling evidence that biological brain architecture contains optimized design principles for cognitive computation:

**Cortical Network Blueprints.** The study demonstrates that the connectivity patterns of human cortical networks, when used as architectural templates for artificial neural networks, improve accuracy by 20% across all evaluated benchmarks. This improvement is consistent across vision, language, and reasoning tasks.

**Evolutionary Optimization.** The improvement from using brain-derived architectures suggests that evolution has optimized neural connectivity for general cognitive computation over hundreds of millions of years. These optimizations are not captured by current architecture search methods, which optimize for specific task distributions.

**Transfer of Biological Design Principles.** The 20% improvement across all benchmarks indicates that biological architectural principles are not task-specific but represent general computational advantages. This supports the hypothesis that biological architecture encodes fundamental computational principles that transcend specific tasks.

**Implications for BIO-NN.** BIO-NN enables systematic investigation of which biological architectural principles drive performance gains. The framework's modular design allows researchers to independently vary architectural components derived from different brain regions and study their individual and synergistic contributions.

### 2.5 Sparse Winner-Take-All Biological Preprocessing (PNAS 2025)

The sparse Winner-Take-All (sWTA) work (PNAS, 2025) demonstrates that biological preprocessing mechanisms provide significant robustness benefits:

**Biological Preprocessing for ViTs.** The study shows that sparse winner-take-all preprocessing—a mechanism directly inspired by biological lateral inhibition and sparse coding—improves Vision Transformer out-of-distribution robustness by 20%. This improvement comes from the preprocessing stage, not from modifying the Transformer architecture itself.

**Out-of-Distribution Generalization.** The 20% improvement in OOD robustness is particularly significant because it addresses one of the most critical limitations of current AI systems: the inability to generalize beyond training distributions. Biological sparse coding mechanisms appear to enforce a form of regularization that promotes robust feature learning.

**Mechanism.** The sWTA mechanism implements competitive dynamics where only the most strongly activated neurons survive, suppressing weak activations. This produces sparse, high-contrast representations that are more robust to noise and distributional shifts.

**Implications for BIO-NN.** BIO-NN's sparse computation module implements sWTA as a configurable preprocessing mechanism. The framework enables investigation of how sWTA interacts with other biological mechanisms to produce robust, generalizable representations.

### 2.6 CH-HNN: Hybrid ANN-SNN with Biological E/I Balance (Nature Communications 2025)

CH-HNN (Nature Communications, 2025) demonstrates that hybrid ANN-SNN architectures with biological constraints can significantly outperform conventional approaches:

**Excitatory-Inhibitory Balance.** CH-HNN implements biologically realistic excitatory-inhibitory balance, where excitatory and inhibitory populations maintain approximate balance. This balance is not merely a constraint but a computational feature that enables stable, efficient information processing.

**Homeostatic Adaptation.** The architecture includes homeostatic mechanisms that dynamically adjust neuronal excitability to maintain stable activity levels. This prevents the runaway excitation or silencing that can occur in deep spiking networks.

**Temporal Coding.** CH-HNN exploits temporal coding—the precise timing of spikes—rather than relying solely on rate coding. Temporal coding enables more efficient information transmission and more precise temporal computations.

**Performance Gains.** CH-HNN achieves 10%+ improvement over standard CNNs on benchmark tasks, demonstrating that biological constraints can improve rather than limit artificial network performance.

**Implications for BIO-NN.** CH-HNN validates BIO-NN's approach of integrating E/I balance, homeostatic adaptation, and temporal coding as core architectural features. BIO-NN provides a systematic framework for investigating how these mechanisms interact and scale.

### 2.7 Neuromorphic Hardware: SpiNNaker2

SpiNNaker2 (2025) represents the current state of the art in neuromorphic hardware, providing critical infrastructure for scaling biological computation:

**Scale.** SpiNNaker2 supports 150,000 neurons with 1.8 billion synaptic events per second. While still far from brain-scale simulation, this represents a significant advance toward biologically realistic neural computation.

**Efficiency.** The entire system operates at just 1 watt of power consumption. This is approximately 20 million times more energy-efficient than simulating equivalent neural networks on conventional GPUs. The efficiency arises from event-driven computation—synaptic events are processed only when they occur, rather than processing all connections at every timestep.

**Real-Time Operation.** SpiNNaker2 achieves real-time neural simulation, enabling investigation of temporal dynamics at biologically realistic timescales.

**Implications for BIO-NN.** BIO-NN includes a neuromorphic backend module that can target SpiNNaker2 and similar platforms. This enables researchers to validate computational models on real neuromorphic hardware and study the interaction between algorithmic and hardware-level biological fidelity.

### 2.8 Complementary Learning Systems and Continual Learning

The Complementary Learning Systems (CLS) framework (McClelland et al., 1995) remains foundational for understanding biological continual learning. Recent work has extended CLS in several directions relevant to BIO-NN:

**Hippocampal-Cortical Interaction.** Computational models of hippocampal-cortical interaction demonstrate how fast hippocampal learning and slow cortical consolidation can support lifelong learning without catastrophic interference (Kirkpatrick et al., 2017; van de Ven et al., 2020).

**Sleep-Dependent Consolidation.** Models of sleep-dependent memory consolidation show how replay and restructuring during offline periods can stabilize newly learned information (Lewis & Durrant, 2011; Wei et al., 2019).

**Synaptic Consolidation.** The synaptic tagging and capture hypothesis (Frey & Morris, 1997) provides a molecular mechanism for selective stabilization of recently modified synapses, a key requirement for continual learning.

**Implications for BIO-NN.** BIO-NN's emergence module integrates CLS-inspired mechanisms, including replay, consolidation, and sleep-like offline processing, as components of the broader superintelligence mechanism investigation.

### 2.9 Excitatory-Inhibitory Balance and Network Stability

The balance between excitatory and inhibitory populations is a fundamental organizing principle of biological neural circuits. Recent work has established:

**E/I Balance as a Computational Feature.** Rather than merely constraining network dynamics, E/I balance enables specific computational operations including gain control, contrast normalization, and temporal filtering (Okun & Lampl, 2008).

**Homeostatic Regulation of E/I Balance.** The brain maintains E/I balance through homeostatic mechanisms that dynamically adjust excitatory and inhibitory synaptic strengths (Turrigiano, 2012). Disruption of E/I balance is implicated in numerous neurological and psychiatric conditions.

**E/I Balance in Artificial Networks.** Recent work has shown that incorporating E/I balance constraints in artificial neural networks can improve training stability and generalization (Song et al., 2020).

**Implications for BIO-NN.** BIO-NN implements E/I balance as a configurable constraint that interacts with other biological mechanisms. The framework enables investigation of how E/I balance contributes to the critical dynamical properties associated with superintelligence.

### 2.10 Scale-Free Networks and Critical Phenomena

Scale-free network topology—where the degree distribution follows a power law—has been observed throughout the brain and is hypothesized to support critical dynamics:

**Scale-Free Connectivity in the Brain.** Cortical networks exhibit scale-free connectivity patterns at multiple scales, from local microcircuits to long-range cortical connections (Eguíluz et al., 2005).

**Criticality and Computation.** Systems at criticality exhibit maximal dynamic range, optimal information transmission, and scale-free correlations (Beggs & Plenz, 2003). These properties are computationally advantageous for information processing.

**Homeostatic Maintenance of Criticality.** Biological networks appear to use homeostatic mechanisms to maintain themselves near criticality, adjusting synaptic strengths and excitability to stay in the critical regime (Shew et al., 2015).

**Implications for BIO-NN.** BIO-NN's scale-free routing module and dynamical phase analysis toolkit enable systematic investigation of how scale-free topology and criticality contribute to superintelligent computation.

---

## 3. The BIO-NN Framework

### 3.1 Design Principles

BIO-NN is designed around principles derived from the 2025-2026 literature on biological superintelligence:

**Modularity.** Each biological mechanism is implemented as an independent module that can be enabled, disabled, or modified without affecting other components. This enables systematic ablation studies and investigation of mechanism interactions.

**Multi-Scale Integration.** BIO-NN supports investigation at three scales: microscale (individual neuron dynamics), mesoscale (circuit-level interactions and E/I balance), and macroscale (network-level emergent properties and dynamical phases).

**Dynamical Phase Awareness.** Unlike prior frameworks, BIO-NN explicitly tracks whether the system exhibits the critical dynamical properties associated with superintelligence. The framework includes tools for measuring criticality, spectral properties, and scale-free correlations.

**Hardware-Software Co-Design.** BIO-NN supports both software simulation and neuromorphic hardware deployment, enabling investigation of how hardware-level biological fidelity affects computational properties.

**Safety and Interpretability.** As biologically inspired systems become more capable, understanding their emergent properties—including potential failure modes—becomes critical. BIO-NN includes modules for safety analysis and interpretability.

### 3.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        BIO-NN Framework                         │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│   Neuron    │   Circuit    │   Network    │    Emergence &     │
│   Models    │   Dynamics   │   Scale      │    Safety          │
├─────────────┼──────────────┼──────────────┼────────────────────┤
│ • LIF       │ • E/I Balance│ • Scale-Free │ • Criticality      │
│ • DualLIF   │ • STDP       │   Routing    │   Detection        │
│ • Adaptive  │ • Homeostatic│ • Hybrid     │ • Spectral         │
│   LIF       │   Plasticity │   Attention  │   Analysis         │
│ • Izhikevich│ • Structural │ • Dynamic    │ • Interpretability │
│ • AdEx      │   Plasticity │   Topology   │ • Safety Metrics   │
│ • DEXAT     │ • Temporal   │ • Brain-     │ • Anomaly          │
│   Encoding  │   Coding     │   Derived    │   Detection        │
│             │              │   Architect. │                    │
├─────────────┴──────────────┴──────────────┴────────────────────┤
│                    Dynamical Phase Analysis                      │
│  • Collective Criticality  • Spectral Condensation             │
│  • Homeostatic Protection  • Scale-Free Correlations           │
│  • Manifold Stability      • Reentrant Mixing                  │
├─────────────────────────────────────────────────────────────────┤
│                    Neuromorphic Backend                          │
│  • SpiNNaker2  • Loihi  • TrueNorth  • Software Simulation    │
├─────────────────────────────────────────────────────────────────┤
│                    Configuration & Experiment Management         │
│  • JSON Config  • Ablation System  • Result Tracking           │
│  • Reproducibility  • Hardware Abstraction                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Neuron Models

BIO-NN implements a comprehensive library of neuron models spanning the range from computationally efficient to biologically detailed:

#### 3.3.1 Leaky Integrate-and-Fire (LIF)

The basic LIF model serves as the foundation:

```
τ_m * dV/dt = -(V - V_rest) + R * I(t)
if V >= V_th:
    spike = True
    V = V_reset
```

Parameters: membrane time constant (τ_m), resting potential (V_rest), threshold (V_th), reset potential (V_reset), membrane resistance (R).

#### 3.3.2 DualLIF

Inspired by SpikingBrain2.0, the DualLIF model maintains separate excitatory and inhibitory membrane potentials:

```
τ_exc * dV_exc/dt = -(V_exc - V_rest) + R_exc * I_exc(t)
τ_inh * dV_inh/dt = -(V_inh - V_rest) + R_inh * I_inh(t)
V_eff = V_exc - α * V_inh
if V_eff >= V_th:
    spike = True
    V_exc = V_reset_exc
    V_inh = V_reset_inh
```

The DualLIF model captures the antagonistic interaction between excitatory and inhibitory inputs within individual neurons, enabling richer temporal coding than standard LIF.

#### 3.3.3 Adaptive LIF

Extends LIF with adaptation currents:

```
τ_m * dV/dt = -(V - V_rest) + (I - w_adapt) / C_m
dw_adapt/dt = -w_adapt / τ_w + a * spike
V_th_eff = V_th + w_adapt
```

#### 3.3.4 Izhikevich Model

Captures diverse biological firing patterns with computational efficiency:

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
if v >= 30:
    v = c
    u = u + d
```

Parameters a, b, c, d control firing patterns (regular spiking, fast spiking, chattering, intrinsically bursting, etc.).

#### 3.3.5 Adaptive Exponential Integrate-and-Fire (AdEx)

Combines exponential spike initiation with adaptation:

```
τ_m * dV/dt = -(V - V_rest) + Δ_T * exp((V - V_th)/Δ_T) + R * I(t) - w
dw/dt = a(V - V_rest) - w
if V >= V_peak:
    V = V_reset
    w = w + b
```

#### 3.3.6 DEXAT Encoding

Inspired by SpikingBrain2.0's Dual Excitatory-Antagonistic Transformation:

```
Input decomposition:
    x_exc = max(x, 0)       # Excitatory component
    x_inh = max(-x, 0)      # Antagonistic component

Spike generation:
    spike_exc = generate_spikes(x_exc, rate_or_temporal)
    spike_inh = generate_spikes(x_inh, rate_or_temporal)

Reconstruction:
    output = decode(spike_exc) - decode(spike_inh)
```

DEXAT enables more efficient spike generation than rate coding by preserving the sign and magnitude information in continuous-valued inputs through excitatory-antagonistic decomposition.

### 3.4 Circuit-Level Dynamics

#### 3.4.1 Excitatory-Inhibitory Balance

BIO-NN implements biologically realistic E/I balance at the circuit level:

**Dale's Law Compliance.** Each neuron is either excitatory or inhibitory (or modulatory), consistent with Dale's law. Excitatory neurons release glutamate or other excitatory neurotransmitters; inhibitory neurons release GABA or glycine.

**Balance Monitoring.** The framework continuously monitors E/I balance through the ratio of excitatory to inhibitory synaptic weights and currents. Target E/I ratios can be configured based on experimental data from specific brain regions.

**Homeostatic E/I Regulation.** When E/I balance deviates from target values, homeostatic mechanisms adjust synaptic strengths or excitability to restore balance. This prevents pathological states (runaway excitation, complete silencing) while maintaining computational flexibility.

**Diverse Inhibitory Populations.** BIO-NN supports multiple inhibitory interneuron types (parvalbumin-positive, somatostatin-positive, vasoactive intestinal peptide-positive) with distinct computational properties, reflecting the biological diversity of inhibitory circuits.

#### 3.4.2 Spike-Timing-Dependent Plasticity (STDP)

BIO-NN implements multiple STDP variants:

**Basic STDP:**
```
Δw = A_+ * exp(-Δt/τ_+)  if Δt > 0 (pre before post)
Δw = -A_- * exp(Δt/τ_-)  if Δt < 0 (post before pre)
```

**Triplet STDP** (Pfister & Gerstner, 2006): Captures interactions between multiple spikes for more accurate modeling of experimental STDP data.

**Voltage-Dependent STDP** (Clopath et al., 2010): Incorporates post-synaptic membrane potential into the plasticity rule, enabling bidirectional plasticity based on postsynaptic depolarization.

**Modulated STDP:** Three-factor learning rules where synaptic modification depends on pre-synaptic activity, post-synaptic activity, and a global neuromodulatory signal (dopamine, acetylcholine, etc.).

#### 3.4.3 Homeostatic Plasticity

BIO-NN implements multiple homeostatic mechanisms:

**Synaptic Scaling:** Multiplicative adjustment of all incoming synaptic weights to maintain a target firing rate:
```
w_i(t+1) = w_i(t) * (target_rate / actual_rate)^η
```

**Intrinsic Plasticity:** Adjustment of neuronal excitability (threshold, adaptation) to maintain target activity levels.

**Metaplasticity:** Modification of plasticity rules based on activity history, implementing the BCM rule (Bienenstock, Cooper, & Munro, 1982).

**E/I Homeostatic Regulation:** Dynamic adjustment of excitatory and inhibitory strengths to maintain target E/I balance and overall network stability.

#### 3.4.4 Structural Plasticity

**Synapse Growth (Synaptogenesis):** New synapses form based on neuronal proximity and activity correlations, with growth probability dependent on pre-synaptic and post-synaptic activity.

**Synapse Elimination (Pruning):** Weak or inactive synapses are removed based on weight magnitude, activity correlation, or age.

**Neurogenesis:** New neurons can be added to the network, with integration rules determining connectivity.

**Activity-Dependent Remodeling:** Structural changes are driven by activity patterns, implementing Hebbian-like structural plasticity.

### 3.5 Network-Scale Mechanisms

#### 3.5.1 Scale-Free Adaptive Routing

Inspired by Dragon Hatchling and biological scale-free connectivity:

**Adaptive Path Selection.** Information routes through the network via dynamically determined paths based on content. Different inputs recruit different computational pathways, enabling efficient resource allocation.

**Scale-Free Topology.** The network implements scale-free connectivity where some neurons serve as hubs with many connections while most neurons have few connections. This topology supports critical dynamics and efficient information transmission.

**Adaptive Computation Depth.** Different inputs recruit different amounts of computation. Simple inputs are processed shallowly; complex inputs recruit deeper processing. This enables O(N) average-case complexity while maintaining the capacity for deep processing when needed.

**Content-Dependent Routing.** Rather than fixed feedforward pathways, information flows through the network based on content, enabling flexible, task-dependent computation.

#### 3.5.2 Hybrid Attention Mechanisms

Inspired by SpikingBrain2.0 and brain-as-architectural-prior:

**Spiking Attention.** Attention mechanisms implemented with spiking neurons, enabling temporal coding of attention weights and efficient sparse attention computation.

**Brain-Derived Connectivity Templates.** Architectural templates derived from human cortical connectivity patterns (as demonstrated by the 20% improvement from brain-as-architectural-prior).

**Cross-Modal Attention.** Attention mechanisms that operate across different modalities (visual, auditory, linguistic) using biologically plausible mechanisms.

**Temporal Attention.** Attention that operates over temporal sequences using spike-timing-dependent mechanisms, enabling precise temporal binding.

#### 3.5.3 Dynamic Topology

**Adaptive Connectivity.** Network connectivity adapts based on task demands, with new connections forming for novel tasks and unused connections being pruned.

**Community Structure.** Networks develop modular community structure with dense intra-module connections and sparse inter-module connections, reflecting cortical modularity.

**Small-World Properties.** Networks maintain small-world properties (high clustering, short path lengths) that support efficient local and global information processing.

### 3.6 Emergence and Safety Analysis

#### 3.6.1 Criticality Detection

BIO-NN includes tools for characterizing whether a system exhibits critical dynamics:

**Branching Ratio Analysis.** Measurement of the branching ratio (average number of downstream neurons activated by a single upstream neuron) to determine proximity to criticality.

**Power-Law Detection.** Statistical tests for power-law distributions in avalanche size, duration, and other quantities, indicating scale-free dynamics.

**Correlation Analysis.** Measurement of spatial and temporal correlations across the network to characterize collective dynamics.

**Spectral Analysis.** Eigenvalue spectrum analysis to detect spectral condensation and reentrant mixing.

#### 3.6.2 Spectral Analysis

**Eigenvalue Spectrum Tracking.** Continuous monitoring of the eigenvalue spectrum of the network's dynamics to detect phase transitions and spectral condensation.

**Manifold Dimension Estimation.** Estimation of the dimensionality of the stabilized manifold on which dynamics occur.

**Spectral Gap Analysis.** Measurement of the spectral gap (separation between largest and sub-largest eigenvalues) as an indicator of computational coherence.

#### 3.6.3 Safety Metrics

**Behavioral Stability.** Monitoring for emergent behaviors that deviate from expected patterns, including oscillatory instabilities, chaotic dynamics, and representation collapse.

**Distributional Shift Detection.** Detection of when the system encounters inputs significantly different from its training distribution, triggering appropriate responses (uncertainty estimation, exploration, fallback to simpler processing).

**Capability Monitoring.** Tracking of emergent capabilities as the system scales, with tools for detecting capability thresholds and phase transitions.

**Interpretability Tools.** Visualization and analysis tools for understanding what the system has learned and how it processes information, including activation visualization, connectivity analysis, and causal intervention tools.

### 3.7 Neuromorphic Backend

BIO-NN supports deployment on neuromorphic hardware platforms:

**SpiNNaker2 Backend.** Direct compilation and deployment to SpiNNaker2 hardware, with automatic mapping of BIO-NN neuron models and connectivity patterns to hardware resources.

**Intel Loihi Backend.** Support for Loihi neuromorphic processors, with optimization for Loihi's specific computational architecture.

**IBM TrueNorth Backend.** Support for TrueNorth neural synapse chips.

**Software Simulation Backend.** High-performance GPU-accelerated simulation for researchers without access to neuromorphic hardware. Supports both exact spike simulation and approximate methods for scaling to larger networks.

**Hardware Abstraction Layer.** A unified API that abstracts hardware-specific details, enabling researchers to develop models in software and deploy to hardware without code changes.

### 3.8 Configuration System

BIO-NN uses a hierarchical JSON-based configuration system:

```json
{
  "framework_version": "2.0",
  "neuron_models": {
    "excitatory": {
      "type": "dual_lif",
      "params": {
        "tau_exc": 20.0,
        "tau_inh": 10.0,
        "v_rest": -65.0,
        "v_th": -50.0,
        "alpha": 0.3
      }
    },
    "inhibitory": {
      "type": "adaptive_lif",
      "params": {
        "tau_m": 15.0,
        "v_rest": -65.0,
        "v_th": -50.0,
        "tau_w": 100.0,
        "a": 0.01,
        "b": 0.5
      }
    }
  },
  "circuit_dynamics": {
    "ei_balance": {
      "enabled": true,
      "excitatory_fraction": 0.8,
      "homeostatic_regulation": true
    },
    "stdp": {
      "enabled": true,
      "variant": "triplet",
      "a_plus": 0.01,
      "a_minus": 0.012
    },
    "homeostatic": {
      "synaptic_scaling": true,
      "intrinsic_plasticity": true,
      "target_rate": 0.05,
      "time_scale": 1000.0
    }
  },
  "network_scale": {
    "scale_free_routing": {
      "enabled": true,
      "gamma": 2.5,
      "adaptive_computation": true
    },
    "hybrid_attention": {
      "enabled": true,
      "type": "spiking_attention",
      "brain_derived_template": true
    },
    "dynamic_topology": {
      "enabled": true,
      "growth_rate": 0.001,
      "pruning_threshold": 0.01
    }
  },
  "emergence_safety": {
    "criticality_detection": true,
    "spectral_analysis": true,
    "safety_metrics": true,
    "interpretability": true
  },
  "neuromorphic_backend": {
    "target": "software_simulation",
    "hardware_target": "spinnaker2",
    "precision": "float32"
  },
  "dynamical_phase": {
    "monitor_criticality": true,
    "track_spectral_condensation": true,
    "measure_scale_free_correlations": true,
    "homeostatic_protection_monitor": true
  }
}
```

### 3.9 Experiment Management

**Experiment Tracking.** Every experiment is assigned a unique identifier with full configuration, code version, random seeds, and hardware configuration recorded.

**Dynamical Phase Logging.** Continuous monitoring and logging of criticality metrics, spectral properties, and scale-free correlations throughout training and inference.

**Ablation System.** Systematic ablation tools that enable researchers to independently vary each biological mechanism and measure its contribution to dynamical phase properties and computational performance.

**Comparison Framework.** Tools for comparing multiple experimental conditions across metrics, including statistical significance testing and effect size calculations.

**Reproducibility.** Full experiment reproducibility through configuration tracking, random seed management, and dependency recording.

---

## 4. Research Directions

BIO-NN is designed to address fundamental questions about biological superintelligence mechanisms. We organize these questions into six research directions:

### 4.1 What Makes Biological Intelligence Superintelligent?

**Question:** What specific combination of biological mechanisms gives rise to the computational capabilities that define biological superintelligence?

**Approach with BIO-NN:** Systematic ablation of biological mechanisms while monitoring criticality metrics and computational performance. By independently varying STDP, homeostatic plasticity, E/I balance, sparse coding, structural plasticity, and scale-free routing, we can identify which mechanisms are necessary and sufficient for superintelligent-level computation.

**Expected Insights:** Identification of a minimal set of biological mechanisms required for superintelligent computation, and characterization of how these mechanisms interact synergistically.

### 4.2 How Does the Brain Maintain Criticality?

**Question:** What mechanisms enable the brain to maintain itself near criticality, and how does this criticality contribute to computational capability?

**Approach with BIO-NN:** Use the dynamical phase analysis toolkit to characterize criticality in networks with different combinations of homeostatic mechanisms. Test whether specific homeostatic mechanisms (synaptic scaling, intrinsic plasticity, metaplasticity) are specifically adapted for maintaining criticality.

**Expected Insights:** Characterization of which homeostatic mechanisms are critical for maintaining the superintelligent dynamical phase, and how these mechanisms interact to provide robust criticality maintenance.

### 4.3 How Do Biological Architectures Encode Computational Principles?

**Question:** What computational principles are encoded in biological brain architecture, and how can these principles be transferred to artificial systems?

**Approach with BIO-NN:** Use brain-derived architectural templates (from brain-as-architectural-prior) as starting configurations, then systematically vary architectural components to identify which biological design principles drive performance gains. Compare architectures derived from different brain regions (cortex, hippocampus, cerebellum) to understand region-specific computational principles.

**Expected Insights:** A catalog of biological architectural principles with their computational contributions, enabling principled design of brain-inspired artificial architectures.

### 4.4 How Do Scale-Free Dynamics Support Computation?

**Question:** How do scale-free dynamics and adaptive computation enable efficient, powerful computation?

**Approach with BIO-NN:** Implement Dragon Hatchling-inspired scale-free routing with varying degrees of biological fidelity. Measure how scale-free dynamics affect computational efficiency, representation quality, and generalization. Characterize the relationship between scale-free topology, criticality, and computational performance.

**Expected Insights:** Understanding of how scale-free dynamics enable adaptive computation, and how biological scale-free mechanisms differ from engineered scale-free architectures.

### 4.5 How Does E/I Balance Contribute to Superintelligence?

**Question:** What is the specific computational contribution of excitatory-inhibitory balance to superintelligent computation?

**Approach with BIO-NN:** Vary E/I balance parameters systematically while monitoring criticality, spectral properties, and computational performance. Test whether E/I balance is required for criticality maintenance, spectral condensation, or other properties of the superintelligent dynamical phase.

**Expected Insights:** Characterization of the computational role of E/I balance in superintelligent computation, including its interaction with homeostatic plasticity and criticality.

### 4.6 How Can Neuromorphic Hardware Support Superintelligent Computation?

**Question:** How does hardware-level biological fidelity affect the computational properties of biologically inspired networks?

**Approach with BIO-NN:** Deploy identical models to software simulation, SpiNNaker2, and other neuromorphic platforms. Compare criticality metrics, spectral properties, and computational performance across hardware targets. Identify hardware-level biological features that are critical for superintelligent computation.

**Expected Insights:** Understanding of which hardware-level biological features are essential for superintelligent computation, informing both neuromorphic hardware design and algorithm development.

---

## 5. Expected Contributions

### 5.1 Theoretical Contributions

1. **A dynamical phase framework for biological superintelligence.** BIO-NN provides the first systematic computational framework for investigating superintelligence as a dynamical phase, enabling rigorous testing of the theoretical predictions from (arxiv: 2602.08483).

2. **Mechanism interaction characterization.** Systematic identification of how biological mechanisms interact to produce superintelligent computation, including synergistic, redundant, and antagonistic interactions.

3. **Criticality maintenance mechanisms.** Identification of which biological mechanisms are specifically adapted for maintaining criticality, and how these mechanisms interact.

4. **Brain architecture principles.** A catalog of biological architectural principles and their computational contributions, informed by systematic investigation using brain-derived templates.

### 5.2 Methodological Contributions

5. **A multi-scale investigation methodology.** A systematic methodology for investigating biological mechanisms across scales, from single neurons to network-level emergent properties.

6. **Dynamical phase analysis toolkit.** Open-source tools for characterizing criticality, spectral properties, and scale-free dynamics in neural networks.

7. **Ablation methodology for biological mechanisms.** A rigorous methodology for isolating the contributions of individual biological mechanisms and their interactions.

8. **Hardware-software co-investigation methodology.** Methods for investigating how hardware-level biological fidelity affects computational properties.

### 5.3 Empirical Contributions

9. **Characterization of biological mechanism contributions.** Empirical measurements of the individual and synergistic contributions of biological mechanisms to computational performance and criticality.

10. **Scaling laws for biological mechanisms.** Characterization of how the contributions of biological mechanisms change as network size, task complexity, and training duration increase.

11. **Hardware-specific insights.** Empirical measurements of how different neuromorphic hardware platforms affect the computational properties of biologically inspired networks.

12. **Safety characterization.** Empirical characterization of emergent properties, failure modes, and safety-relevant behaviors of biologically inspired networks at different scales.

### 5.4 Practical Contributions

13. **A reference implementation.** An open-source, well-documented framework that enables researchers across neuroscience, machine learning, and neuromorphic engineering to investigate biological superintelligence mechanisms.

14. **A community benchmark.** Standardized benchmarks and evaluation protocols for biological superintelligence research, enabling fair comparison across approaches.

15. **Design principles for brain-inspired AI.** Actionable design principles derived from BIO-NN investigations, informing the development of more capable and efficient artificial intelligence systems.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

**Scale limitations.** While BIO-NN supports investigation at multiple scales, current software simulation is limited to networks of approximately 10^6 neurons. Brain-scale simulation (10^11 neurons) remains beyond reach, though neuromorphic hardware backends partially address this limitation.

**Biological fidelity.** BIO-NN's neuron models, while spanning a range of biological detail, remain simplified compared to actual biological neurons. The framework supports multiple fidelity levels, but the most detailed biological mechanisms (e.g., detailed dendritic computation, glial cell interactions, neurotransmitter dynamics) are not yet implemented.

**Experimental validation.** Many of BIO-NN's biological mechanisms are based on computational neuroscience models that may not accurately capture biological reality. Results from BIO-NN should be interpreted as computational hypotheses that require experimental validation.

**Task complexity.** Initial benchmarks focus on relatively simple tasks. Investigating superintelligent-level computation on complex, real-world tasks remains a significant challenge.

**Safety analysis.** While BIO-NN includes safety metrics, comprehensive safety analysis of biologically inspired systems at scale requires additional development, particularly for detecting and mitigating emergent risks.

### 6.2 Future Directions

**Brain-scale simulation.** As neuromorphic hardware scales to larger neuron counts, BIO-NN will support investigation at biologically realistic scales. Collaboration with neuromorphic hardware teams (SpiNNaker, Loihi) will enable this scaling.

**Glial cell integration.** Biological neural computation involves not only neurons but also glial cells (astrocytes, oligodendrocytes, microglia) that modulate synaptic transmission, metabolism, and network dynamics. Future versions of BIO-NN will incorporate glial cell models.

**Detailed dendritic computation.** Biological dendrites perform complex nonlinear computations that are not captured by point neuron models. Future BIO-NN versions will support multi-compartment neuron models with detailed dendritic computation.

**Closed-loop interaction.** Future BIO-NN versions will support closed-loop interaction with environments, enabling investigation of embodied biological intelligence and active inference.

**Theoretical framework development.** BIO-NN will be used to develop and test theoretical frameworks for understanding biological superintelligence, including information-theoretic, dynamical systems, and category-theoretic approaches.

**Clinical applications.** Understanding biological superintelligence mechanisms has implications for treating neurological and psychiatric conditions where these mechanisms are disrupted. Future BIO-NN versions will include clinical application modules.

---

## 7. Conclusion

The question of what makes biological intelligence superintelligent is now tractable. The convergence of theoretical advances (superintelligence as a dynamical phase), experimental findings (brain-as-architectural-prior, sWTA, CH-HNN), engineering achievements (SpikingBrain2.0, Dragon Hatchling, SpiNNaker2), and computational frameworks (BIO-NN) creates an unprecedented opportunity to investigate the biological mechanisms underlying superintelligence.

BIO-NN addresses a critical gap in this investigation: the lack of a unified, modular, multi-scale framework for systematically studying how biological mechanisms interact to produce superintelligent computation. By integrating large-scale spiking neurons, hybrid attention, scale-free routing, homeostatic regulation, neuromorphic backends, and dynamical phase analysis within a configurable architecture, BIO-NN enables the controlled experiments needed to understand biological superintelligence.

The framework is designed not merely to replicate biological mechanisms but to understand them—to isolate individual contributions, characterize mechanism interactions, and identify the minimal set of biological principles required for superintelligent computation. This understanding has direct implications for building more capable, more efficient, and safer artificial intelligence systems.

The 2025-2026 literature demonstrates that biological computational principles—sparsity, temporal coding, homeostatic regulation, scale-free connectivity, excitatory-inhibitory balance—are not merely bio-inspired decorations but core architectural innovations that drive performance. BIO-NN provides the systematic framework needed to understand why these principles work, how they interact, and how they can be transferred to artificial systems.

As biologically inspired systems become more capable, understanding their emergent properties—including potential failure modes—becomes critical. BIO-NN's safety and interpretability modules address this need, enabling responsible investigation of superintelligent biological mechanisms.

The BIO-NN framework is released as open source, with documentation, tutorials, and benchmark tasks to enable broad adoption across neuroscience, machine learning, and neuromorphic engineering communities. We invite researchers to use BIO-NN to investigate the biological mechanisms of superintelligence and to contribute to this emerging field.

---

## References

Abbott, L. F. (1999). Lapicque's introduction of the integrate-and-fire model neurons (1907). *Brain Research Bulletin*, 50(5-6), 303-304.

Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of plasticity. *Trends in Neurosciences*, 19(4), 126-130.

Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

Berridge, C. W., & Waterhouse, B. D. (2003). The locus coeruleus–noradrenergic system: modulation of behavioral state and state-dependent cognitive processes. *Brain Research Reviews*, 42(1), 33-84.

Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982). Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. *Journal of Neuroscience*, 2(1), 32-48.

Bostrom, N. (2014). *Superintelligence: Paths, dangers, strategies*. Oxford University Press.

Bouret, S., & Sara, S. J. (2005). Network reset: a simplified overarching control of integration. *Trends in Cognitive Sciences*, 9(11), 505-510.

Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. *Journal of Neurophysiology*, 94(5), 3637-3642.

Brzosko, Z., Mello-Ribas, J. L., & Bhatt, D. H. (2019). Modulation of spike-timing-dependent plasticity for reinforcement learning. *PLoS Computational Biology*, 15(2), e1006705.

Chklovskii, D. B., Mel, B., & Svoboda, K. (2002). Cortical rewiring and information storage. *Nature*, 416(6882), 881-887.

Clopath, C., Büsing, L., Vasilaki, E., & Gerstner, W. (2010). Voltage-based spike timing-dependent plasticity. *Neural Computation*, 22(11), 2868-2887.

Davies, M., Srinivasa, N., Lin, T. H., Chinya, G., Cao, Y., Choday, S. H., ... & Wang, H. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82-99.

De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., ... & Tuytelaars, T. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3366-3385.

Desai, N. S., Rutherford, L. C., & Turrigiano, G. G. (2002). Plasticity in the intrinsic excitability of cortical pyramidal neurons. *Nature Neuroscience*, 5(6), 527-532.

Eguíluz, V. M., Chialvo, D. R., Cecchi, G. A., Baliki, M., & Apkarian, A. V. (2005). Scale-free brain functional networks. *Physical Review Letters*, 94(1), 018102.

Einarsson, E., & Amari, S. I. (2018). Structural plasticity as a new approach to unsupervised learning. *Neural Computation*, 30(1), 1-24.

Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020). Orthogonal gradient descent for continual learning. In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)* (pp. 3762-3772).

Fauth, M., & van Rossum, M. C. (2019). Self-organized remaintenance: a continuous repair mechanism for synaptic plasticity. *PLoS Computational Biology*, 15(7), e1006513.

Fino, P. F., Glowinski, J., & Deniau, J. M. (2005). Dopamine-dependent facilitation of LTP in prefrontal cortex. *Journal of Neurophysiology*, 94(6), 4172-4183.

French, R. M. (1999). Catastrophic interference in connectionist networks: Can it be predicted, can it be prevented? In *Advances in Neural Information Processing Systems* (pp. 1176-1182).

Frey, U., & Morris, R. G. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533-536.

Funkhouser, J., & Bhatt, D. H. (2021). Synaptic consolidation: an approach to continual learning. *Frontiers in Computational Neuroscience*, 15, 65.

Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single neurons, populations, plasticity*. Cambridge University Press.

Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. *Current Opinion in Neurobiology*, 16(6), 710-715.

Hasselmo, M. E., & Sarter, M. (2011). Modes and models of forebrain cholinergic neuromodulation of cognition. *Neuropsychopharmacology*, 36(1), 52-73.

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.

Hua, J. Y., & Smith, S. J. (2004). Neural activity and the dynamics of central nervous system development. *Nature Neuroscience*, 7(4), 327-332.

Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.

Kanerva, P. (1988). *Sparse distributed memory*. MIT Press.

Kempter, R., Gerstner, W., & van Hemmen, J. L. (1999). Hebbian learning of temporal patterns in spiking networks. *Neural Computation*, 11(7), 1739-1775.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hassabis, D. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems* (pp. 1097-1105).

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

Lennie, P. (2003). The cost of cortical computation. *Current Biology*, 13(6), 493-497.

Lewis, P. A., & Durrant, S. J. (2011). Overlapping memory replay during sleep and waking: from memory consolidation to memory enhancement? *Philosophical Transactions of the Royal Society B: Biological Sciences*, 366(1564), 406-413.

Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935-2947.

London, M., & Häusser, M. (2005). Dendritic computation. *Annual Review of Neuroscience*, 28, 503-532.

Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. In *Advances in Neural Information Processing Systems* (pp. 6467-6476).

Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.

Maex, R., & Orban, G. A. (1996). Spike-timing dependent plasticity in recurrent circuits of the hippocampal region. *Neuroreport*, 7(15-17), 2483-2488.

Marder, E. (2012). Neuromodulation of neuronal circuits: back to the future. *Neuron*, 76(1), 1-11.

Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, 275(5297), 213-215.

Martins, A., & Astudillo, R. (2016). From Softmax to Sparsemax: A sparse model of attention and multi-label classification. In *Proceedings of the 33rd International Conference on Machine Learning* (pp. 1614-1623).

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419-457.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

Merolla, P. A., Arthur, J. V., Alvarez-Icaza, R., Cassidy, A. S., Sawada, J., Akopyan, F., ... & Modha, D. S. (2014). A million spiking-neuron integrated circuit with a scalable communication network and interface. *Science*, 345(6197), 668-673.

Miconi, T., ChaneY, J., & Stanley, K. O. (2018). Synaptic plasticity as neural function approximation. In *Proceedings of the International Conference on Learning Representations*.

Miconi, T., Van Looy, J., & Chollet, F. (2019). Differentiable plasticity: training plastic neural networks with backpropagation. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 4805-4814).

Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological models of synaptic plasticity based on spike timing. *Biological Cybernetics*, 98(6), 459-478.

Mundy, A., Day, J. J., & Bhatt, D. H. (2015). Complementary learning systems for continual learning. *arXiv preprint arXiv:1505.05383*.

Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51-63.

Okun, M., & Lampl, I. (2008). Instantaneous coupling of excitation and inhibition generates balanced input. *Nature Neuroscience*, 11(11), 1290-1292.

Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381(6583), 607-609.

Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54-71.

Pfister, J. P., & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience*, 26(38), 9673-9682.

Poirazi, P., Branco, T., & Bhatt, D. H. (2003). Pyramidal neurons as dendritic integrators. *Trends in Neurosciences*, 26(9), 523-526.

Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2001-2010).

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems* (pp. 2980-2990).

Rumelhart, D. E., & Zipser, D. (1985). Feature discovery by competitive learning. *Cognitive Science*, 9(1), 75-112.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

Scardapane, S., Comminiello, D., Scardapane, S., Uncini, A., & Comminiello, D. (2017). Network sparse coding with adaptive dictionary learning. *IEEE Transactions on Neural Networks and Learning Systems*, 28(9), 2137-2148.

Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.

Shew, W. L., & Plenz, D. (2013). The functional benefits of criticality in the cortex. *The Neuroscientist*, 19(1), 88-100.

Shew, W. L., Yang, H., Petermann, T., Roy, R., & Plenz, D. (2015). Neuronal avalanches imply maximum dynamic range in cortical networks at criticality. *Journal of Neuroscience*, 29(49), 15595-15600.

Shin, H., Lee, J. K., Kim, J., & Kim, I. (2017). Continual learning with deep generative replay. In *Advances in Neural Information Processing Systems* (pp. 2990-2999).

Song, H. F., Liu, G., Wang, X. J., & Pehlevan, C. (2020). Task-dependent gating in spiking neural networks. *arXiv preprint arXiv:2006.11441*.

Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919-926.

Stepanyants, A., Hof, P. R., & Chklovskii, D. B. (2002). Neurogeometry and potential synaptic connectivity. *Journal of Neuroscience*, 22(13), 5700-5711.

Sutton, R. S., & Barto, A. G. (1998). *Reinforcement learning: An introduction*. MIT Press.

Tavanaei, A., Ghodrati, M., Kheradpisheh, S. R., Masquelier, T., & Maida, A. (2019). Deep learning in spiking neural networks. *Neural Networks*, 111, 47-63.

Thrun, S. (1995). A lifelong learning perspective for mobile robot control. In *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 201-214).

Turrigiano, G. G. (1999). Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same. *Trends in Neurosciences*, 22(5), 221-227.

Turrigiano, G. G. (2012). Homeostatic synaptic plasticity: local and global mechanisms for stabilizing neuronal function. *Cold Spring Harbor Perspectives in Biology*, 4(1), a005736.

Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391(6670), 892-896.

van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). Brain-inspired replay for continual learning with artificial neural networks. *Nature Communications*, 11(1), 4069.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

Wei, H., Kuzovkin, A., & Zhou, J. (2019). Sleep-inspired generative replay for continual learning. *arXiv preprint arXiv:1909.09787*.

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.

Yoon, J., Yang, E., Lee, J., & Hwang, S. J. (2018). Lifelong learning with dynamically expandable networks. In *Proceedings of the International Conference on Learning Representations*.

Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks. *Neural Computation*, 30(6), 1514-1541.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 3987-3995).

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. In *Proceedings of the International Conference on Learning Representations*.
