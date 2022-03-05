# Representation Learning for Reinforcement Learning

A curated list of papers that apply representation learning (RepL) in reinforcement learning (RL).

## Why RepL for RL?

A major reason to apply RepL in RL is to solve problems with high-dimensional state-action spaces. Another motivation of applying RepL in RL is to improve the *sample efficiency* problem. Specifically, we usually want to incorporate some *inductive biases*, i.e., structural information, about the tasks/envs into the representations towards better performance.

- Prevalent RL methods requires lots of supervisions.
    - Instead of only learning from reward signals, we can also learn from the collected data.
- Previous methods are sample inefficient in vision-based RL.
    - Good representations can accelerate learning from images.
- Most of current RL agents are task-specific.
    - Good representations can generalize well across different tasks, or adapt quickly to new tasks.
- Effective exploration is challenging in many RL tasks.
    - Good representations can accelerate exploration.

## Challenges

- Sequential data
- Interactive learning tasks

## Methods

Some popular methods of applying RepL in RL.

- Auxiliary tasks, *i.e.*, reconstruction, MI maximization, entropy maximization, dynamics prediction.
  - ACL, APS, AVFs, CIC, CPC, DBC, Dreamer, DreamerV2, DyNE, IDDAC, PBL, PI-SAC, PlaNet, RCRL, SLAC, SAC-AE, SPR, ST-DIM, TIA, UNREAL, Value-Improvement Path, World Model.
- Contrastive learning.
  - ACL, ATC, Contrastive Fourier, CURL, RCRL. 
- Data augmentation.
  - DrQ, DrQ-v2, PSEs, RAD.
- Bisimulation.
  - DBC, PSEs.
- Causal inference.
  - MISA.

## Workshops

- [Self-supervision for Reinforcement Learning @ ICLR 21](https://sslrlworkshop.github.io/)
- [Unsupervised Reinforcement Learning @ ICML 2021](https://urlworkshop.github.io/)

##   Related Work

- Self-Supervised Learning
- Invariant Representation Learning

## Papers

### Vision-based Control

- [[arXiv' 18][CPC] Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- [[NeurIPS' 19][AVFs] A Geometric Perspective on Optimal Representations for Reinforcement Learning](https://arxiv.org/abs/1901.11530)
- [[NeurIPS' 19] Discovery of Useful Questions as Auxiliary Tasks](https://arxiv.org/abs/1909.04607)
- [[NeurIPS' 19][ST-DIM] Unsupervised state representation learning in atari](https://arxiv.org/abs/1906.08226) ([Code](https://github.com/mila-iqia/atari-representation-learning))
- [[NeurIPS' 20][PI-SAC] Predictive Information Accelerates Learning in RL](https://arxiv.org/abs/2007.12401) ([Code](https://github.com/google-research/pisac))
- [[NeurIPS' 20][SLAC] Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Mode](https://arxiv.org/abs/1907.00953) ([Code](https://github.com/alexlee-gk/slac))
- [[NeurIPS' 20][RAD] Reinforcement Learning with Augmented Data](https://arxiv.org/abs/2004.14990) ([Code](https://github.com/MishaLaskin/rad))
- [[ICML' 20][CURL] Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/abs/2004.04136) ([Code](https://www.github.com/MishaLaskin/curl))
- [[ICLR' 20][DynE] Dynamics-aware Embeddings](https://arxiv.org/abs/1908.09357) ([Code](https://github.com/dyne-submission/dynamics-aware-embeddings))
- [[NeurIPS' 21] An Empirical Investigation of Representation Learning for Imitation](https://openreview.net/forum?id=kBNhgqXatI) ([Code](https://github.com/HumanCompatibleAI/eirli))
- [[NeurIPS' 21][SGI] Pretraining Representations for Data-Efficient Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html) ([Code](https://github.com/mila-iqia/SGI))
- [[AAAI' 21][SAC-AE] Improving Sample Efficiency in Model-Free Reinforcement Learning from Images](https://arxiv.org/abs/1910.01741) ([Code](https://sites.google.com/view/sac-ae/home))
- [[AAAI' 21][Value-Improvement Path] Towards Better Representations for Reinforcement Learning](https://arxiv.org/abs/2006.02243)
- [[AISTATS' 21] On The Effect of Auxiliary Tasks on Representation Dynamics](https://arxiv.org/abs/2102.13089)
- [[ICLR' 21][SPR] Data-Efficient RL with Self-Predictive Representations](https://arxiv.org/abs/2007.05929) ([Code](https://github.com/mila-iqia/spr))
- [[ICLR' 21][DBC] Learning invariant representations for reinforcement learning without reconstruction](https://arxiv.org/abs/2006.10742) ([Code](https://github.com/facebookresearch/deep_bisim4control))
- [[ICLR' 21][DrQ] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/abs/2004.13649) ([Code](https://github.com/denisyarats/drq))
- [[ICLR' 21][RCRL] Return-based Contrastive Representation Learning for RL](https://arxiv.org/abs/2102.10960)
- [[ICML' 21][ATC] Decoupling representation learning from reinforcement learning](https://arxiv.org/abs/2009.08319) ([Code](https://github.com/astooke/rlpyt/tree/master/rlpyt/ul))
- [[ICML' 21][APS] Active Pretraining with Successor Features](http://proceedings.mlr.press/v139/liu21b.html)
- [[ICML'21][IDDAC] Decoupling Value and Policy for Generalization in Reinforcement Learning](https://arxiv.org/abs/2102.10330) ([Code](https://github.com/rraileanu/idaac))
- [[ICLR' 22][DrQ-v2] Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://openreview.net/forum?id=_SJ-_yyes8) [(Code)](https://github.com/facebookresearch/drqv2)

### Theory

- [[ICML' 19] DeepMDP: Learning Continuous Latent Space Models for Representation Learning](https://arxiv.org/abs/1906.02736)
- [[ICML' 20] Learning with Good Feature Representations in Bandits and in RL with a Generative Model](https://arxiv.org/abs/1911.07676)
- [[ICLR' 20] Is a good representation sufficient for sample efficient reinforcement learning?](https://arxiv.org/abs/1910.03016)
- [[ICLR' 21] Impact of Representation Learning in Linear Bandits](https://arxiv.org/abs/2010.06531)
- [[arXiv' 21] Model-free Representation Learning and Exploration in Low-rank MDPs](https://arxiv.org/abs/2102.07035)
- [[arXiv' 21] Representation Learning for Online and Offline RL in Low-rank MDPs](https://arxiv.org/abs/2110.04652) :heart:
- [[arXiv' 21] Action-Sufficient State Representation Learning for Control with Structural Constraints](https://arxiv.org/abs/2110.05721)
- [[arXiv' 21] Exponential Lower Bounds for Planning in MDPs With Linearly-Realizable Optimal Action-Value Functions](https://arxiv.org/abs/2010.01374)

### Low-rank MDPs

- [Model-free Representation Learning and Exploration in Low-rank MDPs]
- [FLAMBE: Structural Complexity and Representation Learning of Low Rank MDPs]
- [Representation Learning for Online and Offline RL in Low-rank MDPs]
- [Provably Efficient Representation Learning in Low-rank Markov Decision Processes]

### Offline RL

- [[NeurIPS' 21][Contrastive Fourier] Provable Representation Learning for Imitation with Contrastive Fourier Features](https://arxiv.org/abs/2105.12272) ([Code](https://github.com/google-research/google-research/tree/master/rl_repr))
- [[ICML' 21][ACL] Representation Matters: Offline Pretraining for Sequential Decision Making](https://arxiv.org/abs/2102.05815) ([Code](https://github.com/google-research/google-research/tree/master/rl_repr))
- [[ICML' 21] Instabilities of offline rl with pre-trained neural representation](https://arxiv.org/abs/2103.04947)

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
- [[ICML WS' 21] Representation Learning for Out-of-distribution Generalization in Reinforcement Learning](https://openreview.net/forum?id=I8rHTlfITWC) :heart:
- [[arXiv' 22][CIC] Contrastive Intrinsic Control for Unsupervised Skill Discovery](https://arxiv.org/abs/2202.00161) ([Code](https://sites.google.com/view/cicrl/))
- [[AISTATS' 22] On the Generalization of Representations in Reinforcement Learning](https://arxiv.org/abs/2203.00543) :heart:

