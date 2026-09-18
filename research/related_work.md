# Related Work: Biologically-Inspired Neural Networks for Continual Learning

## 1. Spiking Neural Network Frameworks

### Norse
Norse provides a PyTorch-native SNN library with temporal dynamics and surrogate gradients. Strengths: clean API, strong GPU support, differentiable spiking. Weaknesses: limited biological fidelity beyond membrane dynamics; no built-in structural plasticity. [Nordahl et al., 2021]

### SpikingJelly
Open-source framework with CUDA-accelerated spiking layers. Strengths: wide neuron model support, multi-step simulation, good documentation. Weaknesses: focuses on pattern recognition benchmarks rather than biological plausibility; no neuromodulation. [Fang et al., 2020]

### BindsNET
Built on PyTorch for simulating biologically plausible SNNs on GPU. Strengths: STDP learning, receptive field mapping, modular architecture. Weaknesses: limited scalability; no structural plasticity or neuromodulation. [Hazan et al., 2018]

### NEST
Large-scale neural simulator for point neurons. Strengths: highly accurate biological simulation, massive parallelism, extensive neuron models. Weaknesses: steep learning curve, not GPU-accelerated for deep learning workflows, poor integration with modern ML pipelines. [Gewaltig & Diesmann, 2007]

### Brian2
Python-based simulator with a concise equation-based syntax. Strengths: flexible, well-documented, supports complex neuron models. Weaknesses: CPU-only, slow for large networks, not designed for integration with ANNs. [Stimberg et al., 2019]

### Lava (Intel)
Framework for neuromorphic computing on Intel Loihi. Strengths: hardware-software co-design, real neuromorphic execution, process modules. Weaknesses: locked to Intel hardware ecosystem, limited community adoption, constrained biological modeling flexibility. [Orchard et al., 2022]

### snnTorch
PyTorch-based SNN library focused on accessibility. Strengths: simple API, excellent tutorials, auto-differentiation through time. Weaknesses: minimal biological detail, no structural plasticity, limited to small-scale experiments. [Eisen & Orchard, 2021]

### Comparison Summary

| Framework | Ease of Use | Biological Fidelity | GPU Support | Structural Plasticity | Neuromodulation |
|-----------|------------|---------------------|-------------|----------------------|-----------------|
| Norse | High | Medium | Yes | No | No |
| SpikingJelly | High | Low-Medium | Yes | No | No |
| BindsNET | Medium | Medium-High | Yes | No | No |
| NEST | Low | High | Limited | Partial | No |
| Brian2 | Medium | High | No | No | No |
| Lava | Medium | Low | Loihi only | No | No |
| snnTorch | High | Low | Yes | No | No |

**What has been done**: Mature ecosystems for simulating spiking neurons with surrogate gradients and GPU acceleration.

**What hasn't been done**: No single framework integrates structural plasticity, synaptic plasticity, and neuromodulation in a configurable, user-friendly system.

---

## 2. Bio-Inspired Continual Learning

### Miconi et al. 2018
"Differentiable Plasticity" introduced differentiable Hebbian plasticity for continual learning in ANNs. Each synapse has a learnable plasticity coefficient alongside its static weight. Results on changing MNIST tasks showed improved plasticity over fixed networks. **Limitation**: Only synaptic plasticity is modeled; no structural changes or neuromodulation. [Miconi et al., 2018]

### Hazan et al. 2018
Applied BindsNET's STDP-based learning to continual learning benchmarks. Demonstrated that SNNs with STDP can learn sequential tasks with experience replay. **Limitation**: Replay buffers are not biologically plausible; no structural adaptation. [Hazan et al., 2018]

### Koyama et al. 2020
Investigated weight consolidation and synaptic tagging in SNNs for continual learning. Proposed bio-inspired regularization to protect important weights. **Limitation**: Focused on synaptic mechanisms only; structural plasticity absent. [Koyama et al., 2020]

### Tax & Lanora (recent work)
Explored meta-learning approaches for continual learning in spiking networks, using learned learning rules. **Limitation**: Requires separate meta-training phase; not fully online continual learning. [Tax et al., 2023]

### Comparison

| Approach | Structural Plasticity | Synaptic Plasticity | Neuromodulation | Continual Learning |
|----------|----------------------|---------------------|-----------------|-------------------|
| Miconi et al. | No | Hebbian | No | Yes |
| Hazan et al. | No | STDP | No | Yes |
| Koyama et al. | No | STDP + consolidation | No | Yes |
| Tax & Lanora | No | Meta-learned | No | Yes |
| **BIO-NN** | **Yes** | **STDP** | **Yes** | **Yes** |

**What has been done**: Several approaches combine STDP with replay or regularization for continual learning.

**What hasn't been done**: Joint structural + synaptic + neuromodulatory adaptation in a single continual learning framework.

---

## 3. Structural Plasticity in ANNs/SNNs

### Network Pruning (Han et al. 2015)
"Deep Compression" demonstrated that large networks can be pruned post-training to remove redundant weights and neurons. Achieves 9-13x compression with minimal accuracy loss. **Biological gap**: Pruning is a one-time post-training step, not a dynamic process during learning. [Han et al., 2015]

### Neural Architecture Search (NAS)
Automated methods (ENAS, DARTS, ProxylessNAS) discover optimal architectures. Strengths: can find efficient architectures. Weaknesses: computationally expensive, no biological basis, requires full training data access. [Pham et al., 2018; Liu et al., 2019]

### Dynamic Networks
BranchyNet, BlockDrop, and conditional computation allow different network paths for different inputs. Strengths: adaptive computation. Weaknesses: routing learned via backpropagation, not activity-dependent growth/pruning. [Teerapittayanon et al., 2017]

### Growing Neural Networks
ProgressiveNet and PackNet add capacity during training. Strengths: allow increasing model capacity. Weaknesses: capacity additions are predetermined, not driven by local activity rules. [Rusu et al., 2016; Mallya & Lazebnik, 2018]

### Biological Structural Plasticity
In biology, synaptogenesis and synapse elimination occur continuously, driven by correlated activity. Neurons can sprout new dendrites and form new connections throughout life.

**What has been done**: Pruning and dynamic architectures exist but are fundamentally different from biological structural plasticity (one-shot, global optimization vs. continuous, local rules).

**What hasn't been done**: Activity-dependent structural plasticity driven by local STDP-like rules in SNNs for continual learning.

---

## 4. Neuromodulation in Artificial Systems

### Reward-Modulated STDP (R-STDP)
Modulates STDP learning rates based on global reward signals. Implemented in robotic control tasks with SNNs. Strengths: biologically inspired credit assignment. Weaknesses: reward signal is typically sparse and delayed, leading to credit assignment challenges. [Frémaux & Gerstner, 2016]

### Neuromodulated ANNs
Modulated plasticity layers where global signals (dopamine analogs) gate learning. Strengths: can selectively strengthen or weaken learning. Weaknesses: modulatory signals often hand-crafted rather than learned. [Miconi et al., 2019]

### Global Learning Signals
Frameworks using global error signals to modulate local learning rules. Strengths: biologically more plausible than pure backpropagation. Weaknesses: still rely on gradient-based updates rather than true Hebbian mechanisms. [Lillicrap et al., 2016]

**What has been done**: Simple reward-modulated STDP for robotic control; modulated plasticity in small networks.

**What hasn't been done**: Neuromodulatory systems that jointly control structural plasticity, synaptic plasticity, and network homeostasis in a continual learning context.

---

## 5. Dendritic Computation in ANNs

### Dendritic Networks (Ulyanov et al. 2016)
"Deep Convolutional Networks with Branchy Architecture" explored branching paths inspired by dendritic computation. **Limitation**: Branches are fixed after training, not adaptive. [Ulyanov et al., 2016]

### Attention Mechanisms as Dendritic Computation
Transformer attention can be viewed as dendritic gating, where attention weights modulate information flow like dendritic branches. **Limitation**: Attention is computed globally, not at the individual dendrite level. [Vaswani et al., 2017]

### Capsule Networks
Encode hierarchical relationships with dynamic routing. Strengths: capture part-whole relationships. Weaknesses: high computational cost, limited scalability, routing algorithm not biologically grounded. [Sabour et al., 2017]

### Multicompartment Neuron Models
More biologically accurate models that simulate individual dendritic compartments. Strengths: capture nonlinear dendritic computation. Weaknesses: computationally expensive, rarely integrated into deep learning. [Poirazi et al., 2003]

**What has been done**: Approximate dendritic computation through branching architectures and attention mechanisms.

**What hasn't been done**: Integration of simplified dendritic computation into SNNs for continual learning, balancing biological plausibility with computational tractability.

---

## 6. Predictive Processing in ML

### Prediction Error Networks
Architectures that minimize prediction error across hierarchical levels. Strengths: biologically motivated. Weaknesses: training instability, limited to specific architectures. [Rao & Ballard, 1999]

### Autoencoders as Predictive Coding
Denoising and variational autoencoders can be viewed as implementations of predictive coding, minimizing reconstruction error. Strengths: mature, well-understood. Weaknesses: not explicitly implementing hierarchical prediction error minimization. [Vincent et al., 2008]

### Hierarchical Predictive Coding
Models that implement explicit prediction error propagation across layers. Strengths: biologically grounded. Weaknesses: difficult to scale, training challenges. [Whittington & Bogacz, 2017]

**What has been done**: Theoretical foundations and small-scale implementations of predictive coding in neural networks.

**What hasn't been done**: Practical integration of predictive coding into SNN continual learning frameworks at scale.

---

## 7. Sparse Neural Networks

### Sparse Training (Gale et al. 2019)
Magnitude-based pruning during training achieves sparse networks that train efficiently. Strengths: significant parameter reduction. Weaknesses: sparsity is structural, not dynamic or activity-dependent. [Gale et al., 2019]

### Conditional Computation
Mixture of Experts (MoE) routes inputs to specialized subnetworks. Strengths: increases effective capacity without proportional compute increase. Weaknesses: routing via softmax is not biologically plausible. [Shazeer et al., 2017]

### Event-Driven Computation
SNNs naturally implement event-driven sparsity, only processing non-zero spikes. Strengths: energy-efficient on neuromorphic hardware. Weaknesses: sparse activation is incidental, not learned or structured. [Maass, 1997]

### Lottery Ticket Hypothesis
Dense networks contain sparse subnetworks that can achieve full accuracy when trained in isolation. Strengths: reveals inherent sparsity. Weaknesses: finding lottery tickets requires full training and iterative pruning. [Frankle & Carlin, 2019]

**What has been done**: Magnitude-based pruning, MoE routing, and event-driven computation in SNNs.

**What hasn't been done**: Activity-dependent sparse connectivity driven by local STDP rules with neuromodulatory gating, achieving both energy efficiency and continual learning.

---

## Summary: How BIO-NN Differs

| Dimension | Existing Work | BIO-NN |
|-----------|--------------|--------|
| Structural plasticity | Post-training pruning or predetermined growth | Activity-dependent, local rules during training |
| Synaptic plasticity | STDP in isolation | STDP + structural + neuromodulatory co-adaptation |
| Neuromodulation | Simple reward modulation | Multi-signal modulation controlling all plasticity |
| Continual learning | Replay buffers or regularization | Structural adaptation enables replay-free continual learning |
| Evaluation | Single mechanism, single benchmark | Systematic ablation across mechanisms and benchmarks |
| Interpretability | Black-box or partial visualization | Live visualization of all internal dynamics |
