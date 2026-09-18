# BIO-NN: A Modular Framework for Investigating Biologically Inspired Mechanisms in Continual Learning

**Authors:** [Author Names]

**Affiliations:** [Institutional Affiliations]

**Corresponding Author:** [Email]

**Venue Target:** Frontiers in Computational Neuroscience / NeurIPS Workshop on Continual Learning / IJCNN

---

## Abstract

Catastrophic forgetting remains one of the most significant obstacles to deploying neural networks in real-world settings where continuous adaptation is required. While biological neural systems demonstrate remarkable capacity for continual learning throughout an organism's lifetime, the mechanisms underlying this ability remain incompletely understood in computational terms. In this paper, we introduce BIO-NN, a modular framework for systematically investigating biologically inspired mechanisms that may mitigate catastrophic forgetting in artificial neural networks. The framework integrates four principal biological mechanisms: spike-timing-dependent plasticity (STDP) and homeostatic synaptic plasticity for adaptive weight updates, structural plasticity for dynamic network topology modification, sparse computation for energy-efficient activation patterns, and neuromodulatory signals for task-dependent modulation of learning. BIO-NN provides a configurable architecture that allows researchers to isolate, combine, and evaluate these mechanisms under controlled experimental conditions. We present a systematic experimental methodology for evaluating both individual contributions and synergistic interactions among biological mechanisms. The framework supports multiple neuron models including standard leaky integrate-and-fire (LIF) and adaptive LIF variants, multiple encoding schemes for converting static data into spike trains, and a flexible experiment management system for reproducible research. We propose hypotheses regarding the potential contributions of each mechanism to continual learning performance and outline a comprehensive experimental protocol spanning standard benchmarks including Split MNIST, Permuted MNIST, and multi-task classification scenarios. BIO-NN aims to bridge the gap between computational neuroscience insights and practical continual learning solutions by providing an accessible, well-documented, and extensible platform for bio-inspired learning research.

**Keywords:** catastrophic forgetting, continual learning, biologically inspired neural networks, spike-timing-dependent plasticity, structural plasticity, sparse computation, neuromodulation, spiking neural networks

---

## 1. Introduction

### 1.1 The Catastrophic Forgetting Problem

Artificial neural networks have achieved remarkable success across a wide range of tasks, from image classification to natural language processing (LeCun et al., 2015; Krizhevsky et al., 2012; Vaswani et al., 2017). However, a fundamental limitation persists: when trained sequentially on multiple tasks, neural networks exhibit catastrophic forgetting—the tendency to lose knowledge of previously learned information upon learning new information (McCloskey & Cohen, 1989; French, 1999). This stands in stark contrast to biological neural systems, which demonstrate remarkable capacity for continual learning throughout an organism's lifetime. A human being can learn thousands of new concepts, skills, and memories over decades without systematically erasing previously acquired knowledge.

The catastrophic forgetting problem manifests in various ways. When a neural network trained on task A is subsequently trained on task B, its performance on task A typically degrades substantially, sometimes reverting to near-chance levels (McCloskey & Cohen, 1989). This occurs because standard gradient-based learning algorithms modify network weights in ways that are optimized for the current task without regard for the preservation of previously learned representations. The权重 updates that improve performance on new tasks interfere with the synaptic configurations that encoded prior knowledge.

In practical terms, catastrophic forgetting imposes significant constraints on the deployment of neural networks in real-world scenarios. Systems that must adapt to new data streams, learn from user interactions over time, or operate in non-stationary environments require either complete retraining on all previously encountered data or acceptance of degraded performance on older tasks (Delange et al., 2021). Both solutions carry substantial costs: retraining is computationally expensive and may violate data privacy constraints, while degraded performance is unacceptable in safety-critical applications.

### 1.2 Biological Continual Learning as Inspiration

Biological neural systems offer compelling evidence that continual learning is achievable. The human brain continuously integrates new information throughout life—learning languages, acquiring skills, forming new memories, and adapting to changing environments—without experiencing the catastrophic interference observed in artificial systems (Abraham & Robins, 2005). While biological learning is not perfect (humans do forget), the degree of knowledge retention achieved over years and decades of learning far exceeds what current artificial systems can accomplish.

The mechanisms underlying biological continual learning are multifaceted and operate across multiple scales, from molecular processes at individual synapses to network-level reorganization. At the synaptic level, biological synapses exhibit complex plasticity rules that go far beyond the simple additive weight updates used in standard artificial neural networks. Spike-timing-dependent plasticity (STDP) adjusts synaptic strengths based on the precise temporal relationships between pre-synaptic and post-synaptic spikes (Bi & Poo, 1998; Markram et al., 1997), enabling Hebbian learning with temporal specificity. Homeostatic plasticity mechanisms maintain neuronal excitability within functional ranges, preventing runaway excitation or silencing (Turrigiano, 1999; Turrigiano et al., 1998). Structural plasticity allows the physical connectivity of neural circuits to change over time, with new synapses forming and existing ones being eliminated (Chklovskii et al., 2002; Stepanyants et al., 2002).

At the network level, biological neural systems exhibit sparse activation patterns, where only a small fraction of neurons are active at any given time (Olshausen & Field, 1996). This sparsity not only reduces energy consumption but may also provide computational benefits by reducing interference between representations. Additionally, neuromodulatory systems—diffuse projections from specialized nuclei that release neurotransmitters such as dopamine, acetylcholine, and norepinephrine—can globally modulate synaptic plasticity, attention, and learning rates in a task-dependent manner (Bouret & Sara, 2005; Marder, 2012).

### 1.3 Potential Mechanisms for Mitigating Forgetting

Several specific biological mechanisms have been proposed as potential contributors to continual learning capability:

**Spike-Timing-Dependent Plasticity (STDP):** This learning rule adjusts synaptic weights based on the relative timing of pre-synaptic and post-synaptic spikes. When a pre-synaptic neuron fires shortly before a post-synaptic neuron, the synapse is strengthened (long-term potentiation), while the reverse timing leads to weakening (long-term depression) (Bi & Poo, 1998). STDP provides a temporally precise, local learning rule that may enable more selective modification of synaptic connections, potentially reducing interference with previously learned representations.

**Structural Plasticity:** Unlike standard neural networks with fixed architectures, biological neural circuits can modify their physical connectivity. New synapses can form between previously unconnected neurons, existing synapses can be eliminated, and new neurons can be integrated into circuits (Chklovskii et al., 2002). Structural plasticity may enable the creation of new capacity for learning new tasks while preserving existing circuit configurations.

**Sparse Computation:** Biological neural systems exhibit sparse activation patterns, where neurons are typically silent and only become active when processing relevant stimuli (Olshausen & Field, 1996). Sparsity can reduce catastrophic forgetting through multiple mechanisms: reducing interference between task representations, improving signal-to-noise ratios, and enabling more efficient allocation of neural resources.

**Neuromodulation:** Diffuse neuromodulatory systems can globally regulate synaptic plasticity, effectively controlling when and how strongly learning occurs (Bouret & Sara, 2005). Task-dependent neuromodulatory signals could potentially gate plasticity to protect consolidated representations while enabling learning of new information.

### 1.4 The Gap: Mechanism Isolation and Integration

Despite significant progress in understanding individual biological mechanisms, a major gap persists in the field: most computational studies investigate one mechanism at a time, making it difficult to understand how multiple mechanisms interact and contribute to continual learning when combined (Tavanaei et al., 2019; Zenke & Ganguli, 2018). Furthermore, the relative contributions of different mechanisms to continual learning performance remain poorly characterized.

Some studies have examined combinations of biological mechanisms, but these efforts typically lack the systematic methodology needed to isolate individual contributions and understand synergistic interactions. For example, a study combining STDP with structural plasticity may demonstrate improved performance, but it cannot determine whether the improvement is attributable to STDP, structural plasticity, or their interaction without careful ablation studies.

The absence of a systematic, modular framework for investigating biological mechanisms in continual learning represents a significant barrier to progress. Researchers wishing to study multiple mechanisms must either develop their own implementations—introducing potential inconsistencies and limiting comparability—or rely on existing frameworks that may not support the specific combinations of mechanisms of interest.

### 1.5 Contributions

In this paper, we introduce BIO-NN, a modular framework for systematically investigating biologically inspired mechanisms in continual learning. Our contributions are:

1. **A modular architecture** that enables independent and combined evaluation of synaptic plasticity (STDP, homeostatic), structural plasticity (synapse growth, pruning), sparse computation, and neuromodulatory mechanisms.

2. **A configurable neuron model system** supporting multiple biologically plausible neuron models, including leaky integrate-and-fire (LIF) and adaptive LIF variants, with adjustable parameters for controlling biological fidelity.

3. **Multiple encoding schemes** for converting static datasets into spike train representations, enabling evaluation of rate-based and temporal coding strategies.

4. **A systematic experimental methodology** for evaluating individual and combined contributions of biological mechanisms to continual learning, including hypotheses, baselines, ablation conditions, and evaluation metrics.

5. **An experiment management system** supporting reproducible research with automated configuration tracking, result logging, and analysis pipelines.

6. **A comprehensive experimental protocol** designed to test specific hypotheses about the contributions of biological mechanisms to continual learning, with benchmarks spanning standard continual learning scenarios.

### 1.6 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work in spiking neural networks, biological plasticity mechanisms, continual learning, and bio-inspired approaches. Section 3 describes the BIO-NN framework architecture, including neuron models, plasticity mechanisms, and the configuration system. Section 4 details the experimental setup, including hypotheses, datasets, baselines, and evaluation protocols. Section 5 presents expected results and placeholders for experimental findings. Section 6 discusses implications, limitations, and future directions. Section 7 concludes the paper.

---

## 2. Related Work

### 2.1 Spiking Neural Networks

Spiking neural networks (SNNs) represent a more biologically plausible approach to neural computation compared to traditional artificial neural networks. Unlike standard neurons that communicate through continuous-valued activations, spiking neurons communicate through discrete spike events, incorporating temporal dynamics into neural computation (Maass, 1997; Gerstner & Kistler, 2002).

The leaky integrate-and-fire (LIF) neuron model is one of the most widely used spiking neuron models due to its computational simplicity and biological plausibility (Abbott, 1999). The LIF model describes the membrane potential dynamics of a neuron as:

τ_m * dV/dt = -(V - V_rest) + R * I(t)

where V is the membrane potential, V_rest is the resting potential, R is the membrane resistance, I(t) is the input current, and τ_m is the membrane time constant. When the membrane potential reaches a threshold V_th, the neuron emits a spike and the potential is reset.

Adaptive LIF models extend the basic LIF model by incorporating adaptation currents that modify the effective threshold based on recent spike history (Brette & Gerstner, 2005). This adaptation enables neurons to exhibit frequency adaptation, a common feature of biological neurons where the firing rate decreases in response to sustained input.

Recent work has explored more detailed neuron models, including the Izhikevich model (Izhikevich, 2003), which can reproduce a wide variety of biological firing patterns with computationally efficient equations, and dendritic models that capture the computational properties of dendritic trees (Poirazi et al., 2003; London & Häusser, 2005).

SNNs have demonstrated competitive performance on various benchmarks, particularly when combined with event-driven computation and specialized hardware (Davies et al., 2018; Merolla et al., 2014). However, training SNNs remains challenging due to the non-differentiability of spike events, leading to the development of surrogate gradient methods (Neftci et al., 2019; Zenke & Ganguli, 2018) and alternative training approaches.

### 2.2 Spike-Timing-Dependent Plasticity

Spike-timing-dependent plasticity (STDP) describes a form of synaptic plasticity where the change in synaptic strength depends on the precise temporal relationship between pre-synaptic and post-synaptic spikes (Bi & Poo, 1998; Markram et al., 1997). The basic STDP rule can be described as:

Δw = A_+ * exp(-Δt/τ_+) if Δt > 0 (pre before post)
Δw = -A_- * exp(Δt/τ_-) if Δt < 0 (post before pre)

where Δt = t_post - t_pre is the spike timing difference, A_+ and A_- are the learning rate amplitudes for potentiation and depression, and τ_+ and τ_- are the respective time constants.

STDP has been studied extensively in both experimental and computational neuroscience. Experimental studies have characterized STDP in various brain regions, including the hippocampus (Bi & Poo, 1998), neocortex (Markram et al., 1997), and striatum (Fino et al., 2005). Computational studies have demonstrated that STDP can support unsupervised feature learning (Kempter et al., 1999), temporal sequence learning (Maex & Orban, 1996), and competitive learning dynamics (Song et al., 2000).

In the context of continual learning, STDP offers several potential advantages. Its local nature—depending only on information available at the synapse—may reduce interference with previously learned representations compared to global optimization methods. The temporal specificity of STDP may enable more selective modification of synaptic connections. However, STDP alone may not be sufficient for complex pattern learning, and its interaction with other plasticity mechanisms is not well understood (Morrison et al., 2008).

Several variants of STDP have been proposed, including triplet STDP (Pfister & Gerstner, 2006), which captures interactions between multiple spikes, and voltage-dependent STDP (Clopath et al., 2010), which incorporates the post-synaptic membrane potential into the plasticity rule. These variants may offer improved biological plausibility and computational capabilities.

### 2.3 Structural Plasticity in Neural Networks

Structural plasticity refers to changes in the physical connectivity of neural circuits, including synapse formation (synaptogenesis), synapse elimination, and neurogenesis (Chklovskii et al., 2002; Stepanyants et al., 2002). Unlike synaptic plasticity, which modifies the strength of existing connections, structural plasticity modifies the network architecture itself.

In biological systems, structural plasticity plays important roles in development, learning, and recovery from injury. Synapse formation and elimination follow activity-dependent rules, with active connections being stabilized and inactive connections being pruned (Hua & Smith, 2004). The balance between synapse formation and elimination maintains overall network connectivity while allowing circuit reorganization.

Several computational models have incorporated structural plasticity mechanisms. Synaptic sampling models (Fauth & van Rossum, 2019) simulate synapse formation and elimination as a stochastic process, with synaptic connections being probabilistically formed or removed based on activity patterns. Network growth models (Einarsson & Amari, 2018) simulate the formation of new synaptic connections based on neuronal proximity and activity correlations.

In the context of continual learning, structural plasticity offers the potential to create new capacity for learning new tasks while preserving existing circuit configurations. By adding new neurons or synapses specifically for encoding new information, structural plasticity may avoid interference with previously learned representations encoded in existing circuitry. However, structural plasticity also increases network size and computational cost, requiring careful management of resource allocation (Chklovskii et al., 2002).

Recent work has explored structural plasticity in artificial neural networks, including dynamic architecture expansion (Yoon et al., 2018) and neural architecture search with structural modifications (Zoph & Le, 2017). However, these approaches typically do not follow biologically realistic rules for structural modification.

### 2.4 Continual Learning

Continual learning (also known as lifelong learning or incremental learning) addresses the challenge of learning from a continuous stream of tasks or data distributions without forgetting previously learned knowledge (Thrun, 1995; Parisi et al., 2019). The field has developed several approaches to mitigate catastrophic forgetting.

**Replay-based methods** store examples from previous tasks and interleave them with new task data during training. Experience Replay (ER) (Rolnick et al., 2019) maintains a buffer of past experiences and samples from this buffer during training. Generative replay approaches (Shin et al., 2017) use generative models to produce pseudo-examples of previous tasks, avoiding the need to store actual examples. These methods can be effective but require memory for storing examples or generative models, and may struggle with large task sequences.

**Regularization-based methods** constrain weight updates to protect important parameters for previous tasks. Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017) uses the Fisher information matrix to estimate parameter importance and applies quadratic penalties to discourage changes to important parameters. Synaptic Intelligence (SI) (Zenke et al., 2017) estimates online parameter importance based on the contribution of each parameter to loss reduction. Learning without Forgetting (LwF) (Li & Hoiem, 2017) uses knowledge distillation to preserve network outputs for previous tasks.

**Architecture-based methods** allocate separate network components for different tasks, either by expanding the network (Rusu et al., 2016) or by routing different tasks through different pathways (Aljundi et al., 2017). These methods can preserve performance on previous tasks but may scale poorly with the number of tasks.

**Optimization-based methods** modify the learning algorithm to reduce interference between tasks. Orthogonal Gradient Descent (OGD) (Farajtabar et al., 2020) projects gradient updates onto subspaces orthogonal to those important for previous tasks. Gradient Episodic Memory (GEM) (Lopez-Paz & Ranzato, 2017) uses episodic memory to constrain gradient updates.

Despite significant progress, continual learning remains challenging, particularly for long task sequences, complex tasks, and scenarios with limited memory or computational resources (De Lange et al., 2021).

### 2.5 Bio-Inspired Continual Learning

A growing body of work has explored biologically inspired approaches to continual learning, drawing on insights from neuroscience to develop more effective learning algorithms.

**Complementary Learning Systems (CLS) theory** (McClelland et al., 1995) proposes that the brain uses two complementary systems: a fast-learning hippocampal system for rapid encoding of new experiences and a slow-learning neocortical system for gradual integration of knowledge. Computational implementations of CLS have demonstrated improved continual learning performance (Kirkpatrick et al., 2017; Mundy et al., 2015).

**Sleep-based consolidation models** draw on the role of sleep in memory consolidation. During sleep, the hippocampus "replays" recent experiences, facilitating their integration into neocortical knowledge structures. Computational models of sleep-dependent consolidation have shown promise for continual learning (Lewis & Durrant, 2011; Wei et al., 2019).

**Synaptic consolidation models** draw on the molecular processes that stabilize synaptic changes over time. The "synaptic tagging and capture" hypothesis (Frey & Morris, 1997) proposes that strong synaptic activation creates tags that capture plasticity-related proteins, enabling long-term stabilization of synaptic changes. Computational models of synaptic consolidation have been combined with continual learning approaches (Funkhouser & Bhatt, 2021).

**Hippocampal replay models** simulate the replay of neural activity patterns observed in the hippocampus during rest and sleep. Replay-based continual learning methods store and replay compressed representations of previous tasks (Shin et al., 2017; van de Ven et al., 2020).

**Cortical-inspired architectures** draw on the hierarchical organization and lateral connectivity of the cerebral cortex. These approaches often incorporate recurrent connections, lateral inhibition, and hierarchical processing (Miconi et al., 2019).

While these bio-inspired approaches have shown promise, most focus on individual mechanisms or specific brain regions. A systematic investigation of multiple biological mechanisms and their interactions remains an open challenge.

### 2.6 Neuromodulation in Artificial Systems

Neuromodulatory systems in the brain—consisting of specialized nuclei that project diffusely throughout the brain and release neurotransmitters such as dopamine, acetylcholine, norepinephrine, and serotonin—play crucial roles in regulating learning, attention, and behavioral flexibility (Bouret & Sara, 2005; Marder, 2012).

Dopamine has been extensively studied in the context of reinforcement learning, where it serves as a reward prediction error signal (Schultz et al., 1997). The temporal difference (TD) learning algorithm (Sutton & Barto, 1998) was directly inspired by observations of dopamine neuron activity. In the context of continual learning, dopamine-dependent plasticity may enable task-dependent modulation of learning rates and synaptic modifications.

Acetylcholine modulates attention and cortical plasticity, enhancing the processing of relevant stimuli while suppressing background activity (Hasselmo & Sarter, 2011). In computational models, acetylcholine-like signals have been used to gate plasticity, controlling when and where synaptic modifications occur (Hasselmo, 2006).

Norepinephrine modulates arousal and behavioral responses to novel or salient stimuli (Berridge & Waterhouse, 2003). In artificial systems, norepinephrine-like signals could potentially regulate exploration-exploitation trade-offs and adaptation to distributional shifts.

Several computational studies have incorporated neuromodulatory mechanisms:

**Modulated STDP:** Studies have explored how dopamine modulation affects STDP, typically by gating plasticity based on reward signals (Brzosko et al., 2019). This creates a three-factor learning rule where synaptic modification depends on pre-synaptic activity, post-synaptic activity, and a global neuromodulatory signal.

**Attention-gated plasticity:** Models incorporating attention-like neuromodulatory signals have shown improved performance on tasks requiring selective learning (Miconi et al., 2018).

**Meta-learning with neuromodulation:** Some approaches use neuromodulatory signals as meta-parameters that control learning dynamics, enabling rapid adaptation to new tasks (Miconi et al., 2019).

Despite these advances, the role of neuromodulation in continual learning is not well understood, and systematic evaluation of neuromodulatory mechanisms in bio-inspired continual learning systems is lacking.

### 2.7 Sparse Computation

Sparse computation—where only a small fraction of neurons are active at any given time—is a fundamental feature of biological neural systems (Olshausen & Field, 1996; Lennie, 2003). In the visual cortex, for example, only 1-5% of neurons are typically active in response to natural stimuli.

Sparsity offers several potential computational advantages:

**Energy efficiency:** Sparse activation reduces the number of computations required, potentially enabling more energy-efficient neural processing. This is particularly relevant for neuromorphic hardware implementations (Davies et al., 2018).

**Representational capacity:** Sparse representations can encode more information per neuron by reducing redundancy and increasing the distinctiveness of neural representations (Olshausen & Field, 1996).

**Reduced interference:** Sparse activation may reduce interference between different task representations by limiting the overlap between active neuron populations (Kanerva, 1988).

**Improved generalization:** Sparse representations may promote better generalization by encouraging the network to learn more robust, distributed representations (Hinton & Salakhutdinov, 2006).

Several mechanisms can induce sparsity in neural networks:

**Lateral inhibition:** Competition between neurons, often implemented through inhibitory interneurons, can ensure that only the most strongly activated neurons fire (Rumelhart & Zipser, 1985).

**Threshold mechanisms:** Setting activation thresholds can ensure that only neurons receiving sufficiently strong input become active (Maass, 1997).

**Regularization:** L1 regularization encourages sparse weight matrices, while L0 regularization directly penalizes non-zero weights (Scardapane et al., 2017).

**Sparse coding:** Algorithms such as sparse coding and independent component analysis learn sparse representations of input data (Olshausen & Field, 1996).

In the context of continual learning, sparsity may reduce catastrophic forgetting by limiting interference between task representations. When different tasks activate different subsets of neurons, learning new tasks is less likely to disrupt representations encoded by neurons involved in previous tasks. However, the interaction between sparsity and other biological mechanisms in continual learning scenarios has not been systematically investigated.

---

## 3. The BIO-NN Framework

### 3.1 Design Principles

BIO-NN is designed around the following core principles:

**Modularity:** Each biological mechanism is implemented as an independent, self-contained module that can be enabled, disabled, or modified without affecting other components. This enables systematic ablation studies and evaluation of individual and combined contributions.

**Biological Plausibility with Practical Flexibility:** The framework provides multiple levels of biological fidelity, from simplified implementations suitable for engineering applications to more detailed models that capture essential biological features. This allows researchers to trade off between biological realism and computational tractability.

**Reproducibility:** The configuration system ensures that every experiment can be exactly reproduced by saving all relevant parameters and random seeds. The experiment management system tracks all results with full provenance information.

**Extensibility:** The framework is designed to be easily extended with new neuron models, plasticity rules, encoding schemes, and experimental protocols. Clear interfaces between components facilitate the addition of new mechanisms.

**Performance:** While biological plausibility is important, the framework is designed to be computationally efficient enough to support experiments with network sizes and task sequences relevant to practical continual learning scenarios.

### 3.2 Architecture Overview

The BIO-NN framework consists of several interconnected components:

#### 3.2.1 Neuron Models

BIO-NN implements multiple neuron models with varying degrees of biological plausibility:

**Leaky Integrate-and-Fire (LIF):** The basic LIF model tracks membrane potential dynamics with a leak toward resting potential. When the potential reaches threshold, the neuron fires and resets:

```
dV/dt = -(V - V_rest) / τ_m + I / C_m
if V >= V_th:
    spike = True
    V = V_reset
```

Parameters include membrane time constant (τ_m), resting potential (V_rest), threshold (V_th), reset potential (V_reset), and membrane capacitance (C_m).

**Adaptive LIF:** Extends the LIF model with an adaptation current that increases the effective threshold following spike activity:

```
dV/dt = -(V - V_rest) / τ_m + (I - w_adapt) / C_m
dw_adapt/dt = -w_adapt / τ_w + a * spike
V_th_eff = V_th + w_adapt
```

The adaptation variable (w_adapt) increases with each spike and decays with time constant τ_w, implementing frequency adaptation.

**Izhikevich Model:** Implements the computationally efficient model that can reproduce diverse biological firing patterns (Izhikevich, 2003):

```
dv/dt = 0.04v^2 + 5v + 140 - u + I
du/dt = a(bv - u)
if v >= 30:
    v = c
    u = u + d
```

Parameters a, b, c, d control the firing pattern (regular spiking, fast spiking, chattering, etc.).

#### 3.2.2 Synaptic Plasticity

**STDP Module:** Implements spike-timing-dependent plasticity with configurable parameters:

- Learning rate amplitudes (A_+, A_-)
- Time constants (τ_+, τ_-)
- Weight bounds (w_min, w_max)
- Variants: basic STDP, triplet STDP, voltage-dependent STDP

The STDP module receives spike timing information and updates synaptic weights accordingly. Homeostatic mechanisms can be combined with STDP to maintain stable network dynamics.

**Homeostatic Plasticity:** Implements mechanisms to maintain neuronal excitability within functional ranges:

- **Synaptic scaling:** Multiplicative adjustment of all incoming synaptic weights to maintain a target firing rate (Turrigiano et al., 1998).
- **Intrinsic plasticity:** Adjustment of neuronal excitability (e.g., threshold, adaptation) to maintain target activity levels (Desai et al., 2002).
- **Metaplasticity:** Modification of plasticity rules based on activity history (Abraham & Bear, 1996).

**Configuration parameters:**
- Target firing rate for homeostatic regulation
- Time scale of homeostatic adjustment
- Strength of homeostatic vs. Hebbian plasticity

#### 3.2.3 Structural Plasticity

The structural plasticity module implements dynamic network topology modification:

**Synapse Growth (Synaptogenesis):**
- New synapses form between neurons based on proximity and activity correlations
- Growth probability depends on pre-synaptic and post-synaptic activity
- Maximum connectivity constraints prevent unbounded growth

**Synapse Elimination (Pruning):**
- Weak or inactive synapses are removed
- Pruning criteria based on weight magnitude, activity correlation, or age
- Scheduled or activity-dependent pruning

**Neuronal Growth (Neurogenesis):**
- New neurons can be added to the network
- Integration rules determine connectivity of new neurons
- Resource constraints limit total network size

**Configuration parameters:**
- Growth rate and criteria for new synapses/neurons
- Pruning threshold and schedule
- Maximum network size constraints
- Connectivity patterns (random, nearest-neighbor, small-world)

#### 3.2.4 Dendritic Computation

BIO-NN supports dendritic computation models that capture the computational properties of biological dendrites (Poirazi et al., 2003):

- **Dendritic compartments:** Each neuron can have multiple dendritic compartments with independent processing
- **Nonlinear dendritic integration:** Dendritic branches can perform nonlinear operations on inputs
- **Dendritic spines:** Individual synapses can be modeled as dendritic spines with independent calcium dynamics

Dendritic computation adds computational capacity to individual neurons, potentially enabling more complex per-neuron computations while maintaining sparse network connectivity.

#### 3.2.5 Sparse Connectivity

The sparse connectivity module implements mechanisms for maintaining sparse activation patterns:

**Lateral Inhibition:**
- Competitive dynamics between neurons
- Winner-take-all or soft competitive inhibition
- Configurable inhibition strength and scope

**Sparse Activation:**
- Top-k activation: Only the k most strongly activated neurons fire
- Threshold-based activation: Neurons fire only when activation exceeds threshold
- Sparsemax: Probabilistic sparse activation (Martins & Astudillo, 2016)

**Sparse Connectivity:**
- Random sparse connectivity patterns
- Structured sparsity (e.g., local connectivity)
- Activity-dependent connectivity modification

**Configuration parameters:**
- Sparsity level (fraction of active neurons)
- Inhibition type and strength
- Connectivity density

#### 3.2.6 Encoding and Decoding

BIO-NN supports multiple encoding schemes for converting static data (e.g., images) into spike trains:

**Rate Coding:** Input values are converted to spike trains with firing rates proportional to input values. Higher input values produce higher firing rates.

**Temporal Coding:** Input values are encoded in spike timing rather than firing rate. Stronger inputs produce earlier spikes (rank-order coding) or more precise spike timing (phase coding).

**Population Coding:** Input values are represented across populations of neurons, with individual neurons having tuning curves centered at different input values.

**Burst Coding:** Input values are encoded in burst patterns, with stronger inputs producing longer or more frequent bursts.

**Decoding Methods:**
- **Population vector decoding:** Linear combination of neural activities weighted by preferred directions
- **Maximum likelihood decoding:** Probabilistic decoding assuming known neural tuning properties
- **Readout layer:** Standard linear or nonlinear readout from spike train representations

**Configuration parameters:**
- Encoding scheme (rate, temporal, population, burst)
- Number of encoding neurons per input dimension
- Spike train duration and resolution
- Decoding method

### 3.3 Configuration System

BIO-NN uses a hierarchical JSON-based configuration system that specifies all aspects of a model and experiment:

```json
{
  "neuron_model": {
    "type": "adaptive_lif",
    "params": {
      "tau_m": 20.0,
      "v_rest": -65.0,
      "v_th": -50.0,
      "v_reset": -65.0,
      "tau_w": 100.0,
      "a": 0.01,
      "b": 0.5
    }
  },
  "plasticity": {
    "stdp": {
      "enabled": true,
      "a_plus": 0.01,
      "a_minus": 0.012,
      "tau_plus": 20.0,
      "tau_minus": 20.0,
      "w_min": 0.0,
      "w_max": 1.0
    },
    "homeostatic": {
      "enabled": true,
      "target_rate": 0.05,
      "time_scale": 1000.0,
      "type": "synaptic_scaling"
    }
  },
  "structural_plasticity": {
    "enabled": true,
    "growth_rate": 0.001,
    "pruning_threshold": 0.01,
    "max_neurons": 1000
  },
  "sparse": {
    "enabled": true,
    "sparsity_level": 0.1,
    "inhibition_type": "lateral",
    "inhibition_strength": 0.5
  },
  "encoding": {
    "scheme": "rate",
    "n_neurons_per_dim": 10,
    "duration": 50.0,
    "dt": 1.0
  },
  "network": {
    "n_inputs": 784,
    "n_hidden": [256, 128],
    "n_outputs": 10,
    "connectivity": "random",
    "connection_probability": 0.1
  }
}
```

The configuration system supports:
- **Validation:** Automatic validation of parameter ranges and combinations
- **Inheritance:** Configurations can inherit from base configurations with overrides
- **Sweeping:** Configuration parameters can be swept for hyperparameter search
- **Versioning:** Configurations are versioned and linked to experimental results

### 3.4 Experiment Management

BIO-NN includes an experiment management system for reproducible research:

**Experiment Tracking:**
- Every experiment is assigned a unique identifier
- Configuration, code version, and random seeds are recorded
- Results are logged with timestamps and metadata

**Result Storage:**
- Metrics are logged at configurable intervals
- Model checkpoints are saved at specified points
- Raw spike data and network states can be optionally recorded

**Analysis Tools:**
- Automated generation of learning curves and comparison plots
- Statistical significance testing across experimental conditions
- Ablation study analysis with effect size calculations

**Reproducibility:**
- Random seeds are managed centrally
- Code version is tracked via git commit hashes
- Dependencies are recorded via package management files

### 3.5 Biological Plausibility Levels

BIO-NN supports multiple levels of biological plausibility to accommodate different research goals:

**Level 1 (Engineering):** Simplified implementations optimized for performance. Includes basic LIF neurons, simplified STDP rules, and approximate structural plasticity. Suitable for engineering applications where biological realism is secondary.

**Level 2 (Inspired):** Biologically inspired mechanisms with practical simplifications. Includes adaptive LIF neurons, standard STDP with homeostatic plasticity, and activity-dependent structural plasticity. Balances biological plausibility with computational efficiency.

**Level 3 (Plausible):** More detailed implementations capturing essential biological features. Includes Izhikevich or adaptive exponential integrate-and-fire (AdEx) neurons (Brette & Gerstner, 2006), triplet STDP, detailed homeostatic mechanisms, and realistic structural plasticity rules. Suitable for computational neuroscience research.

**Level 4 (Detailed):** Highly detailed implementations approximating biological reality. Includes multi-compartment neuron models, biophysically detailed synapse models, and spatially structured connectivity. Computationally expensive but useful for studying detailed biological mechanisms.

Researchers can select the appropriate level based on their research goals, trading off between biological realism and computational tractability.

---

## 4. Experimental Setup

### 4.1 Hypotheses

We propose the following hypotheses regarding the contributions of biological mechanisms to continual learning:

**H0 (Null Hypothesis):** Biologically inspired mechanisms do not significantly improve continual learning performance compared to standard artificial neural network baselines.

**H1 (Plasticity Hypothesis):** STDP and homeostatic plasticity improve continual learning retention by enabling more selective modification of synaptic connections, reducing interference with previously learned representations.

*Rationale:* STDP's temporal specificity may enable more targeted weight updates compared to standard backpropagation, potentially preserving previously learned representations while accommodating new learning. Homeostatic plasticity may maintain stable network dynamics across tasks.

**H2 (Structural Hypothesis):** Structural plasticity improves continual learning by creating new capacity for learning new tasks while preserving existing circuit configurations.

*Rationale:* By adding new neurons or synapses specifically for encoding new information, structural plasticity may avoid interference with existing representations encoded in established circuitry.

**H3 (Sparsity Hypothesis):** Sparse computation improves continual learning by reducing interference between task representations through limited overlap between active neuron populations.

*Rationale:* When different tasks activate different subsets of neurons, learning new tasks is less likely to disrupt representations encoded by neurons involved in previous tasks.

**H4 (Neuromodulation Hypothesis):** Neuromodulatory signals improve continual learning by enabling task-dependent modulation of learning rates and plasticity.

*Rationale:* Task-specific neuromodulatory signals could gate plasticity to protect consolidated representations while enabling learning of new information.

**H5 (Synergy Hypothesis):** Combinations of biological mechanisms produce synergistic improvements in continual learning that exceed the sum of individual contributions.

*Rationale:* Biological systems employ multiple mechanisms simultaneously, suggesting potential synergistic interactions that may not be captured by studying individual mechanisms in isolation.

### 4.2 Datasets

We evaluate BIO-NN on standard continual learning benchmarks:

**MNIST (LeCun et al., 1998):** Handwritten digit classification (0-9) with 60,000 training and 10,000 test images (28×28 pixels). While simple, MNIST serves as a fundamental baseline for evaluating continual learning approaches.

**Fashion-MNIST (Xiao et al., 2017):** Clothing item classification with the same format as MNIST but greater complexity. Provides a more challenging benchmark while maintaining the same input dimensions.

**Split MNIST:** The 10-digit MNIST dataset is split into 5 sequential tasks, each containing 2 digits (e.g., Task 1: digits 0-1, Task 2: digits 2-3, etc.). This benchmark evaluates the network's ability to learn 5 tasks sequentially without forgetting.

**Permuted MNIST:** Each task consists of MNIST images with a fixed random permutation applied to pixel positions. This benchmark evaluates continual learning under input distribution changes, creating a more challenging scenario than Split MNIST.

**Split Fashion-MNIST:** Similar to Split MNIST but using Fashion-MNIST data, providing a more challenging multi-task benchmark.

**Multi-domain scenario:** Combinations of MNIST and Fashion-MNIST tasks, evaluating transfer and interference across related but distinct data distributions.

### 4.3 Baselines

We compare BIO-NN against the following baselines:

**Standard Neural Networks:**
- **MLP:** Standard multi-layer perceptron trained with backpropagation. Provides an upper bound on performance without continual learning mechanisms.
- **MLP + EWC:** MLP with Elastic Weight Consolidation (Kirkpatrick et al., 2017). Represents regularization-based continual learning.

**Spiking Neural Networks:**
- **Basic SNN:** Spiking neural network with LIF neurons trained with surrogate gradients. Provides a baseline for SNN performance without bio-inspired plasticity.
- **SNN + STDP:** Spiking neural network with STDP learning. Evaluates the contribution of STDP in isolation.

**Continual Learning Methods:**
- **iCaRL:** Incremental Classifier and Representation Learning (Rebuffi et al., 2017). Represents exemplar-based continual learning.
- **LwF:** Learning without Forgetting (Li & Hoiem, 2017). Represents knowledge distillation-based continual learning.

**Ablation Baselines:**
- **BIO-NN (full):** Complete BIO-NN framework with all mechanisms enabled.
- **BIO-NN (no plasticity):** BIO-NN without STDP or homeostatic plasticity.
- **BIO-NN (no structural):** BIO-NN without structural plasticity.
- **BIO-NN (no sparse):** BIO-NN without sparse computation mechanisms.

### 4.4 Ablation Conditions

To isolate the contributions of individual mechanisms, we evaluate the following ablation conditions:

**Individual Mechanisms:**
1. **STDP only:** BIO-NN with only STDP enabled (no homeostatic, structural, or sparse mechanisms)
2. **Homeostatic only:** BIO-NN with only homeostatic plasticity enabled
3. **Structural only:** BIO-NN with only structural plasticity enabled
4. **Sparse only:** BIO-NN with only sparse computation mechanisms enabled
5. **Neuromodulation only:** BIO-NN with only neuromodulatory signals enabled

**Paired Mechanisms:**
6. **STDP + Homeostatic:** Evaluates interaction between Hebbian and homeostatic plasticity
7. **STDP + Structural:** Evaluates interaction between synaptic and structural plasticity
8. **STDP + Sparse:** Evaluates interaction between synaptic plasticity and sparsity
9. **Structural + Sparse:** Evaluates interaction between structural plasticity and sparsity

**Higher-order Combinations:**
10. **Plasticity only:** STDP + Homeostatic (no structural or sparse)
11. **Structural only:** Structural + Homeostatic (no STDP or sparse)
12. **All except neuromodulation:** All mechanisms except neuromodulatory signals

**Full Model:**
13. **Full BIO-NN:** All mechanisms enabled

### 4.5 Evaluation Metrics

We evaluate continual learning performance using the following metrics:

**Average Accuracy:** Mean classification accuracy across all tasks after training on the final task:

Acc_avg = (1/T) * Σ acc_i(T)

where acc_i(T) is the accuracy on task i after training on task T tasks.

**Backward Transfer (BWT):** Measures the average change in performance on previous tasks after learning new tasks:

BWT = (1/(T-1)) * Σ [acc_i(T) - acc_i(i)]

Negative BWT indicates catastrophic forgetting, while BWT near zero indicates minimal forgetting.

**Forward Transfer (FWT):** Measures the average change in performance on new tasks due to knowledge from previous tasks:

FWT = (1/(T-1)) * Σ [acc_i(i) - acc_i^baseline(i)]

Positive FWT indicates beneficial transfer, while negative FWT indicates negative transfer.

**Memory Efficiency:** Ratio of memory used by BIO-NN compared to baselines, including parameters, activity buffers, and any replay mechanisms.

**Computational Efficiency:** Number of floating-point operations (FLOPs) or spike events per inference, measuring computational cost.

**Sparsity Level:** Fraction of active neurons during inference, measuring the degree of sparse computation.

**Network Size:** Total number of neurons and synapses, tracking structural plasticity effects over time.

### 4.6 Implementation Details

**Software Stack:**
- Python 3.8+
- PyTorch 2.0+ for tensor operations and GPU acceleration
- Custom CUDA kernels for efficient spike simulation (optional)
- NumPy for numerical operations
- Matplotlib/Seaborn for visualization

**Training Protocol:**
- Task sequence: Tasks presented sequentially in fixed order
- Training epochs per task: [PLACEHOLDER] epochs
- Batch size: [PLACEHOLDER]
- Learning rate: [PLACEHOLDER]
- Evaluation: After each task, evaluate on all seen tasks

**Hardware:**
- Experiments conducted on NVIDIA GPUs (RTX 3090 or equivalent)
- CPU-only fallback for smaller experiments
- Memory: 24GB GPU memory, 64GB system RAM

**Random Seeds:**
- All experiments repeated with [PLACEHOLDER] random seeds
- Results reported as mean ± standard deviation
- Statistical significance tested with paired t-tests or Wilcoxon signed-rank tests

**Code Availability:**
- Code will be released at [PLACEHOLDER repository URL]
- All configurations and trained models will be archived

---

## 5. Results

### 5.1 Single-Task Performance

Before evaluating continual learning, we first assess the ability of BIO-NN to learn individual tasks. Table 1 presents classification accuracy on MNIST and Fashion-MNIST for each model configuration.

**Table 1: Single-task accuracy (%) on MNIST and Fashion-MNIST**

| Model | MNIST | Fashion-MNIST |
|-------|-------|---------------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] |
| SNN + STDP | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no plasticity) | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no structural) | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no sparse) | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] |

*Note: All models have comparable parameter counts. Results represent mean ± std over [PLACEHOLDER] random seeds.*

Expected findings:
- Standard MLP should achieve highest single-task performance due to direct gradient optimization
- SNNs may show slightly lower performance due to spike-based information transmission constraints
- BIO-NN mechanisms may trade off some single-task performance for continual learning benefits

### 5.2 Continual Learning Results

Table 2 presents continual learning performance on Split MNIST (5 tasks) and Permuted MNIST (5 tasks).

**Table 2: Continual learning performance on Split MNIST**

| Model | Avg Acc (%) | BWT (%) | FWT (%) | Final Acc Task 1 (%) |
|-------|-------------|---------|---------|----------------------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| MLP + EWC | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| iCaRL | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| LwF | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no plasticity) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no structural) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (no sparse) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Table 3: Continual learning performance on Permuted MNIST**

| Model | Avg Acc (%) | BWT (%) | FWT (%) | Memory (MB) |
|-------|-------------|---------|---------|-------------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| MLP + EWC | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

Expected findings:
- BIO-NN (full) should show improved average accuracy and reduced forgetting compared to standard baselines
- BWT should be less negative (less forgetting) for BIO-NN configurations
- Performance improvements may be more pronounced on Permuted MNIST due to greater interference

### 5.3 Ablation Study Results

Table 4 presents the abation study results, isolating individual contributions of each biological mechanism.

**Table 4: Ablation study results on Split MNIST (5 tasks)**

| Condition | Avg Acc (%) | ΔAcc vs Full | BWT (%) | ΔBWT vs Full |
|-----------|-------------|--------------|---------|--------------|
| Full BIO-NN | [PLACEHOLDER] | — | [PLACEHOLDER] | — |
| No STDP | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| No Homeostatic | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| No Structural | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| No Sparse | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| No Neuromodulation | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| STDP only | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Homeostatic only | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Structural only | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Sparse only | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Neuromodulation only | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

Expected findings:
- Removing STDP should show the largest drop in performance, supporting H1
- Structural plasticity removal should show moderate impact, supporting H2
- Sparse computation removal should show moderate impact, supporting H3
- Individual mechanisms should show smaller benefits than the full model, supporting H5 (synergy)

### 5.4 Efficiency Analysis

Table 5 compares computational efficiency across model configurations.

**Table 5: Computational efficiency comparison**

| Model | Parameters | FLOPs/Inference | Sparsity (%) | Network Growth |
|-------|------------|-----------------|--------------|----------------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] | 0 | None |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | None |
| BIO-NN (no structural) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | None |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

Expected findings:
- BIO-NN with sparse computation should show reduced active parameters despite potentially larger total network size
- Structural plasticity may increase network size over time, but sparse computation should limit active computation
- Energy efficiency may be improved due to sparse activation patterns

### 5.5 Robustness Analysis

We evaluate robustness to various perturbations:

**Table 6: Robustness to input noise (accuracy under Gaussian noise)**

| Model | Clean | σ=0.1 | σ=0.2 | σ=0.3 |
|-------|-------|-------|-------|-------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Table 7: Robustness to parameter perturbation**

| Model | Clean | 5% perturb | 10% perturb | 20% perturb |
|-------|-------|------------|-------------|-------------|
| MLP | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Basic SNN | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| BIO-NN (full) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

Expected findings:
- BIO-NN may show improved robustness to input noise due to sparse, distributed representations
- Structural plasticity may provide robustness to parameter perturbation through redundancy
- Homeostatic plasticity may help maintain performance under distributional shifts

---

## 6. Discussion

### 6.1 Which Mechanisms Matter Most

[PLACEHOLDER: Discussion of relative contributions of each mechanism based on experimental results]

Based on the ablation study results, we expect to find that:
- STDP provides the strongest individual contribution to continual learning performance
- Structural plasticity provides complementary benefits, particularly for longer task sequences
- Sparse computation improves efficiency with moderate benefits for continual learning
- Neuromodulation may show context-dependent effects, particularly in multi-domain scenarios

### 6.2 Trade-offs Between Mechanisms

[PLACEHOLDER: Discussion of trade-offs between biological mechanisms]

We expect to observe several trade-offs:
- **Performance vs. efficiency:** More biologically detailed models may improve performance but at higher computational cost
- **Single-task vs. continual learning:** Mechanisms optimized for continual learning may slightly reduce single-task performance
- **Plasticity vs. stability:** More plastic networks may learn new tasks faster but forget previous tasks more readily
- **Network size vs. sparsity:** Structural plasticity increases network size, while sparsity reduces active computation

### 6.3 Biological Plausibility vs Performance

[PLACEHOLDER: Discussion of the relationship between biological plausibility and performance]

The BIO-NN framework enables investigation of whether more biologically plausible implementations necessarily lead to better performance. We expect to find that:
- Moderate biological plausibility (Level 2-3) may offer the best trade-off
- Some biological mechanisms may be essential while others can be simplified
- Engineering optimizations may outperform biologically detailed implementations in some scenarios

### 6.4 Limitations

Several limitations of the current study should be acknowledged:

**Dataset complexity:** The benchmarks used (MNIST, Fashion-MNIST) are relatively simple compared to real-world continual learning scenarios. Results may not generalize to more complex datasets such as CIFAR-100, ImageNet, or natural language processing tasks.

**Task sequence:** The experimental setup uses a fixed task sequence with clear task boundaries. Real-world continual learning often involves overlapping tasks, gradual distributional shifts, and no explicit task boundaries.

**Scalability:** The current experiments are limited to relatively small networks and task sequences. Scaling to larger networks and longer task sequences may reveal different patterns of mechanism contributions.

**Biological realism:** While BIO-NN supports multiple levels of biological plausibility, the implementations remain simplified compared to actual biological neural systems. Results should be interpreted with this limitation in mind.

**Evaluation methodology:** The evaluation metrics used (average accuracy, BWT, FWT) may not capture all aspects of continual learning performance. Alternative metrics such as backward transfer efficiency or forward transfer efficiency may provide additional insights.

### 6.5 Failure Analysis

[PLACEHOLDER: Analysis of failure cases and conditions under which mechanisms do not help]

We expect to identify conditions under which biological mechanisms do not improve or even degrade performance:
- Very short task sequences where structural plasticity overhead exceeds benefits
- Tasks requiring dense, overlapping representations where sparsity is detrimental
- Scenarios requiring rapid adaptation where slow homeostatic plasticity is limiting

---

## 7. Conclusion

### 7.1 Summary

In this paper, we presented BIO-NN, a modular framework for systematically investigating biologically inspired mechanisms in continual learning. The framework integrates four principal biological mechanisms—spike-timing-dependent plasticity, structural plasticity, sparse computation, and neuromodulation—within a configurable, extensible architecture.

BIO-NN addresses a critical gap in the field: the lack of a systematic methodology for evaluating individual contributions and synergistic interactions of biological mechanisms in continual learning. By providing independent modules for each mechanism, a hierarchical configuration system, and a comprehensive experimental methodology, BIO-NN enables researchers to conduct controlled ablation studies that isolate the contributions of specific biological mechanisms.

### 7.2 Contributions

Our contributions include:

1. **A modular framework** that enables independent and combined evaluation of biological mechanisms for continual learning, supporting systematic ablation studies and mechanism interaction analysis.

2. **Multiple neuron models and encoding schemes** that provide flexibility in trading off biological plausibility with computational efficiency, accommodating diverse research goals.

3. **A comprehensive experimental methodology** including hypotheses, baselines, ablation conditions, and evaluation metrics designed to characterize the contributions of biological mechanisms to continual learning.

4. **An experiment management system** supporting reproducible research with automated configuration tracking, result logging, and analysis pipelines.

5. **A systematic experimental protocol** designed to test specific hypotheses about biological mechanisms in continual learning, providing a foundation for rigorous empirical investigation.

### 7.3 Future Work

Several directions for future research are suggested by this work:

**Theoretical analysis:** Developing theoretical models to predict when and why specific biological mechanisms should improve continual learning performance. This could include analysis of interference reduction, capacity expansion, and stability-plasticity trade-offs.

**Complex benchmarks:** Evaluating BIO-NN on more challenging continual learning benchmarks, including class-incremental learning, domain-incremental learning, and real-world streaming data scenarios.

**Hardware implementation:** Implementing BIO-NN on neuromorphic hardware platforms such as Intel's Loihi (Davies et al., 2018) or IBM's TrueNorth (Merolla et al., 2014), which could enable more efficient implementation of spiking neural networks and sparse computation.

**Online continual learning:** Extending BIO-NN to support online continual learning scenarios where tasks arrive as continuous data streams without explicit task boundaries.

**Transfer learning:** Investigating whether biologically inspired mechanisms improve transfer learning across related tasks, potentially enabling more efficient multi-task learning.

**Theoretical neuroscience:** Using BIO-NN as a platform for testing computational neuroscience hypotheses about the roles of specific biological mechanisms in learning and memory.

### 7.4 Concluding Remarks

The BIO-NN framework provides a foundation for systematic investigation of biologically inspired mechanisms in continual learning. By bridging computational neuroscience insights and practical continual learning challenges, BIO-NN aims to advance our understanding of how biological principles can inspire more effective artificial learning systems. We hope this framework will facilitate collaborative research across neuroscience, machine learning, and neuromorphic engineering communities, ultimately contributing to the development of artificial systems that can learn continually, efficiently, and robustly—much like biological neural systems.

---

## References

Abbott, L. F. (1999). Lapicque's introduction of the integrate-and-fire model neurons (1907). *Brain Research Bulletin*, 50(5-6), 303-304.

Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of plasticity. *Trends in Neurosciences*, 19(4), 126-130.

Abraham, W. C., & Robins, A. (2005). Retention—where's the elephant in the room? *Hippocampus*, 15(4), 438-444.

Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2017). Memory aware synapses: Learning what (not) to forget. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 139-154).

Berridge, C. W., & Waterhouse, B. D. (2003). The locus coeruleus–noradrenergic system: modulation of behavioral state and state-dependent cognitive processes. *Brain Research Reviews*, 42(1), 33-84.

Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

Bouret, S., & Sara, S. J. (2005). Network reset: a simplified overarching control of integration. *Trends in Cognitive Sciences*, 9(11), 505-510.

Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. *Journal of Neurophysiology*, 94(5), 3637-3642.

Brette, R., & Gerstner, W. (2006). Adaptive exponential integrate-and-fire model. *Journal of Neurophysiology*, 94(5), 3637-3642.

Brzosko, Z., Mello-Ribas, J. L., & Bhatt, D. H. (2019). Modulation of spike-timing-dependent plasticity for reinforcement learning. *PLoS Computational Biology*, 15(2), e1006705.

Chklovskii, D. B., Mel, B., & Svoboda, K. (2002). Cortical rewiring and information storage. *Nature*, 416(6882), 881-887.

Clopath, C., Büsing, L., Vasilaki, E., & Gerstner, W. (2010). Voltage-based spike timing-dependent plasticity. *Neural Computation*, 22(11), 2868-2887.

Davies, M., Srinivasa, N., Lin, T. H., Chinya, G., Cao, Y., Choday, S. H., ... & Wang, H. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82-99.

De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., ... & Tuytelaars, T. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3366-3385.

Desai, N. S., Rutherford, L. C., & Turrigiano, G. G. (2002). Plasticity in the intrinsic excitability of cortical pyramidal neurons. *Nature Neuroscience*, 5(6), 527-532.

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

Shin, H., Lee, J. K., Kim, J., & Kim, I. (2017). Continual learning with deep generative replay. In *Advances in Neural Information Processing Systems* (pp. 2990-2999).

Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919-926.

Stepanyants, A., Hof, P. R., & Chklovskii, D. B. (2002). Neurogeometry and potential synaptic connectivity. *Journal of Neuroscience*, 22(13), 5700-5711.

Sutton, R. S., & Barto, A. G. (1998). *Reinforcement learning: An introduction*. MIT Press.

Tavanaei, A., Ghodrati, M., Kheradpisheh, S. R., Masquelier, T., & Maida, A. (2019). Deep learning in spiking neural networks. *Neural Networks*, 111, 47-63.

Thrun, S. (1995). A lifelong learning perspective for mobile robot control. In *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 201-214).

Turrigiano, G. G. (1999). Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same. *Trends in Neurosciences*, 22(5), 221-227.

Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391(6670), 892-896.

van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). Brain-inspired replay for continual learning with artificial neural networks. *Nature Communications*, 11(1), 4069.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

Wei, H., Kuzovkin, A., & Zhou, J. (2019). Sleep-inspired generative replay for continual learning. *arXiv preprint arXiv:1909.09787*.

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.

Yoon, J., Yang, E., Lee, J., & Hwang, S. J. (2018). Lifelong learning with dynamically expandable networks. In *Proceedings of the International Conference on Learning Representations*.

Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks. *Neural Computation*, 30(6), 1514-1541.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 3987-3995).

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. In *Proceedings of the International Conference on Learning Representations*.
