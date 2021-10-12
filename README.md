# Representation Learning for Reinforcement Learning

A curated list of papers that apply representation learning (RepL) in reinforcement learning (RL).

## Why RepL for RL?

A major motivation of applying RepL in RL is to improve the *sample efficiency* problem. Specifically, we usually want to incorporate some *inductive biases*, i.e., structural information, about the tasks/envs into the representations towards better performance.

- Prevalent RL methods requires lots of supervisions.
    - Instead of only learning from reward signals, we can also learn from the collected data.
- Previous methods are sample inefficient in Vision-based RL.
    - Good representations can accelerate learning from images.
- Most of current RL agents are task-specific.
    - Good representations can generalize across different tasks.
- Effective exploration is challenging in many RL tasks.
    - Good representations can accelerate exploration.

## Methods

Some popular methods of applying RepL in RL.

- Auxiliary tasks, *i.e.*, reconstruction, MI maximization, entropy maximization, dynamics prediction.
  - ACL, APS, AVFs, CPC, DBC, Dreamer, DreamerV2, IDDAC, PBL, PI-SAC, PlaNet, RCRL, SLAC, SAC-AE, SPR, ST-DIM, TIA, UNREAL, Value-Improvement Path, World Model.
- Contrastive learning.
  - ACL, ATC, Contrastive Fourier, CURL, RCRL. 
- Data augmentation.
  - DrQ, PSEs, RAD.
- Bisimulation.
  - DBC, PSEs.
- Causal inference.
  - MISA.

## Workshops

- [Self-supervision for Reinforcement Learning @ ICLR 21](https://sslrlworkshop.github.io/)
- [Unsupervised Reinforcement Learning @ ICML 2021](https://urlworkshop.github.io/)

## Papers

### Vision-based Control

- [[arXiv' 18][CPC] Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [[NeurIPS' 19][AVFs] A Geometric Perspective on Optimal Representations for Reinforcement Learning](https://arxiv.org/abs/1901.11530)
- [[NeurIPS' 19][ST-DIM] Unsupervised state representation learning in atari](https://arxiv.org/abs/1906.08226) ([Code](https://github.com/mila-iqia/atari-representation-learning))
- [[NeurIPS' 20][PI-SAC] Predictive Information Accelerates Learning in RL](https://arxiv.org/abs/2007.12401) ([Code](https://github.com/google-research/pisac))
- [[NeurIPS' 20][SLAC] Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Mode](https://arxiv.org/abs/1907.00953) ([Code](https://github.com/alexlee-gk/slac))
- [[NeurIPS' 20][RAD] Reinforcement Learning with Augmented Data](https://arxiv.org/abs/2004.14990) ([Code](https://github.com/MishaLaskin/rad))
- [[NeurIPS' 21] An Empirical Investigation of Representation Learning for Imitation](https://openreview.net/forum?id=kBNhgqXatI) ([Code](https://github.com/HumanCompatibleAI/eirli))
- [[ICML' 20][CURL] Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/abs/2004.04136) ([Code](https://www.github.com/MishaLaskin/curl))

- [[AAAI' 21][SAC-AE] Improving Sample Efficiency in Model-Free Reinforcement Learning from Images](https://arxiv.org/abs/1910.01741) ([Code](https://sites.google.com/view/sac-ae/home))
- [[AAAI' 21][Value-Improvement Path] Towards Better Representations for Reinforcement Learning](https://arxiv.org/abs/2006.02243)
- [[ICLR' 21][SPR] Data-Efficient RL with Self-Predictive Representations](https://arxiv.org/abs/2007.05929) ([Code](https://github.com/mila-iqia/spr))
- [[ICLR' 21][DBC] Learning invariant representations for reinforcement learning without reconstruction](https://arxiv.org/abs/2006.10742) ([Code](https://github.com/facebookresearch/deep_bisim4control))
- [[ICLR' 21][DrQ] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649) ([Code](https://github.com/denisyarats/drq))
- [[ICLR' 21][RCRL] Return-based Contrastive Representation Learning for RL](https://arxiv.org/abs/2102.10960)
- [[ICML' 21][ATC] Decoupling representation learning from reinforcement learning](https://arxiv.org/abs/2009.08319) ([Code](https://github.com/astooke/rlpyt/tree/master/rlpyt/ul))
- [[ICML' 21][APS] Active Pretraining with Successor Features](http://proceedings.mlr.press/v139/liu21b.html)
- [[ICML'21][IDDAC] Decoupling Value and Policy for Generalization in Reinforcement Learning](https://arxiv.org/abs/2102.10330) ([Code](https://github.com/rraileanu/idaac))

### Theory

- [[ICLR' 21] Impact of Representation Learning in Linear Bandits](https://arxiv.org/abs/2010.06531)

### Offline RL

- [[NeurIPS' 21][Contrastive Fourier] Provable Representation Learning for Imitation with Contrastive Fourier Features](https://arxiv.org/abs/2105.12272) ([Code](https://github.com/google-research/google-research/tree/master/rl_repr))
- [[ICML' 21][ACL] Representation Matters: Offline Pretraining for Sequential Decision Making](https://arxiv.org/abs/2102.05815) ([Code](https://github.com/google-research/google-research/tree/master/rl_repr))

### Model-based RL

- [[NeurIPS' 18][World Model] Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/pdf/1809.01999.pdf)
- [[ICML' 19][PlaNet] Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) ([Code](https://github.com/google-research/planet))
- [[ICLR' 20][Dreamer] Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) ([Code](https://github.com/google-research/dreamer))
- [[ICLR' 21][DreamerV2] Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) ([Code](https://github.com/danijar/dreamerv2))
- [[ICML' 21][TIA] Learning Task Informed Abstractions](https://arxiv.org/abs/2106.15612) ([Code](https://xiangfu.co/tia))

### Multi-task RL

- [[ICLR' 17][UNREAL] Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)
- [[ICML' 20][PBL] Bootstrap latent-predictive representations for multitask reinforcement learning](https://arxiv.org/abs/2004.14646)

### Exploration

- [[NeurIPS' 20] Provably Efficient Exploration for Reinforcement Learning Using Unsupervised Learning](https://arxiv.org/abs/2003.06898) ([Code](https://github.com/FlorenceFeng/StateDecoding))
- [[ICML' 21][RL-Proto] Reinforcement Learning with Prototypical Representations](https://arxiv.org/abs/2102.11271) ([Code](https://github.com/denisyarats/proto))
- [[ICML WS' 21][FittedKDE] Density-Based Bonuses on Learned Representations for Reward-Free Exploration in Deep Reinforcement Learning](https://openreview.net/forum?id=vRSY3L4Rlhp)

### Generalization

- [[ICML' 20][MISA]  Invariant Causal Prediction for Block MDPs](https://arxiv.org/abs/2003.06016) ([Code](https://github.com/facebookresearch/icp-block-mdp))

- [[ICLR' 21][PSEs] Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.05265)





