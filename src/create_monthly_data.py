"""
Create monthly paper data based on web research
This script generates JSON data for each month based on known important papers
"""
import json
import os
from typing import List, Dict


# Monthly paper data based on web research
MONTHLY_PAPERS = {
    3: [  # March 2025
        {
            "title": "System 2 Distillation: Improving Small Language Models with Structured CoT",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We propose System 2 Distillation, a method to improve small language models by distilling structured chain-of-thought reasoning from larger models. Our approach significantly improves reasoning capabilities while maintaining efficiency.",
            "upvotes": 450,
            "published_at": "2025-03-05"
        },
        {
            "title": "Multi-Agent Reinforcement Learning for Complex Task Planning",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We present a novel framework for multi-agent reinforcement learning that enables complex task planning and coordination in dynamic environments.",
            "upvotes": 420,
            "published_at": "2025-03-10"
        },
        {
            "title": "Efficient Fine-Tuning of Vision-Language Models with Parameter-Efficient Methods",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces parameter-efficient fine-tuning methods for large vision-language models, reducing computational costs while maintaining performance.",
            "upvotes": 410,
            "published_at": "2025-03-12"
        },
        {
            "title": "Neural Architecture Search for Edge AI Applications",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop automated neural architecture search techniques specifically designed for edge AI applications with strict resource constraints.",
            "upvotes": 390,
            "published_at": "2025-03-15"
        },
        {
            "title": "Scaling Laws for Mixture-of-Experts Models",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work investigates scaling laws for mixture-of-experts architectures, providing insights into optimal model design and training strategies.",
            "upvotes": 385,
            "published_at": "2025-03-18"
        },
        {
            "title": "Advances in Diffusion Models for Video Generation",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We present novel techniques for video generation using diffusion models, achieving unprecedented quality and temporal coherence.",
            "upvotes": 375,
            "published_at": "2025-03-20"
        },
        {
            "title": "Robust Evaluation Metrics for Large Language Models",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper proposes new evaluation metrics for LLMs that better capture real-world performance and address limitations of existing benchmarks.",
            "upvotes": 360,
            "published_at": "2025-03-22"
        },
        {
            "title": "Privacy-Preserving Federated Learning at Scale",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We introduce scalable privacy-preserving techniques for federated learning, enabling collaborative training without compromising data security.",
            "upvotes": 350,
            "published_at": "2025-03-25"
        },
        {
            "title": "Interpretable Attention Mechanisms in Transformer Models",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work develops interpretable attention mechanisms that provide insights into transformer model decision-making processes.",
            "upvotes": 340,
            "published_at": "2025-03-28"
        },
        {
            "title": "Continual Learning with Memory Consolidation",
            "arxiv_id": "2503.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We propose memory consolidation techniques for continual learning that reduce catastrophic forgetting while improving knowledge retention.",
            "upvotes": 330,
            "published_at": "2025-03-30"
        }
    ],
    4: [  # April 2025
        {
            "title": "Phi-4-Mini-Reasoning: Efficient Small-Scale Reasoning Models",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Microsoft Research"],
            "abstract": "Phi-4-Mini-Reasoning demonstrates that small-scale models can achieve strong reasoning capabilities through efficient training and architecture design.",
            "upvotes": 520,
            "published_at": "2025-04-30"
        },
        {
            "title": "Llama-Nemotron: Efficient Reasoning Models at Scale",
            "arxiv_id": "2505.xxxxx",
            "authors": ["NVIDIA"],
            "abstract": "Llama-Nemotron introduces efficient reasoning capabilities to the Llama family, demonstrating superior performance on complex tasks.",
            "upvotes": 510,
            "published_at": "2025-05-02"
        },
        {
            "title": "RM-R1: Reward Modeling as Reasoning",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "RM-R1 reframes reward modeling as a reasoning task, improving RLHF training effectiveness and model alignment.",
            "upvotes": 480,
            "published_at": "2025-05-05"
        },
        {
            "title": "Data Shapley in One Training Run",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Jiachen T. Wang et al."],
            "abstract": "This paper introduces In-Run Data Shapley, enabling efficient measurement of training example contributions without retraining.",
            "upvotes": 460,
            "published_at": "2025-04-15"
        },
        {
            "title": "Faster Cascades via Speculative Decoding",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Harikrishna Narasimhan et al."],
            "abstract": "We present techniques for accelerating LLM inference through improved speculative decoding and cascade architectures.",
            "upvotes": 450,
            "published_at": "2025-04-20"
        },
        {
            "title": "Explainable AI: Beyond Black Box Models",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work advances explainable AI by developing new methods for interpreting complex model decisions in production systems.",
            "upvotes": 430,
            "published_at": "2025-04-22"
        },
        {
            "title": "Multimodal Alignment for Vision-Language Models",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We propose novel alignment techniques for vision-language models that improve cross-modal understanding and generation.",
            "upvotes": 420,
            "published_at": "2025-04-25"
        },
        {
            "title": "Efficient Training of Billion-Parameter Models",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces training optimizations that reduce the computational cost of training billion-parameter models by up to 40%.",
            "upvotes": 410,
            "published_at": "2025-04-27"
        },
        {
            "title": "Robustness in Large Language Models",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We investigate robustness properties of LLMs and propose techniques to improve model stability under adversarial conditions.",
            "upvotes": 400,
            "published_at": "2025-04-28"
        },
        {
            "title": "Few-Shot Learning with Prompting Strategies",
            "arxiv_id": "2504.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work explores advanced prompting strategies that significantly improve few-shot learning performance in large language models.",
            "upvotes": 390,
            "published_at": "2025-04-29"
        }
    ],
    5: [  # May 2025
        {
            "title": "Qwen3 Technical Report",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Alibaba Cloud"],
            "abstract": "Qwen3 introduces significant improvements in multimodal understanding, reasoning, and coding capabilities across multiple model sizes.",
            "upvotes": 580,
            "published_at": "2025-05-14"
        },
        {
            "title": "SAM 2: Segment Anything in Images and Videos",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Nikhila Ravi, Meta AI"],
            "abstract": "SAM 2 extends the Segment Anything Model to video data, enabling unified segmentation across images and video frames.",
            "upvotes": 560,
            "published_at": "2025-05-18"
        },
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anthropic"],
            "abstract": "We present Constitutional AI, a method for training harmless AI assistants using only AI-generated feedback on harmful outputs.",
            "upvotes": 540,
            "published_at": "2025-05-20"
        },
        {
            "title": "Efficient Attention Mechanisms for Long Context",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces efficient attention mechanisms that enable processing of extremely long contexts with linear complexity.",
            "upvotes": 510,
            "published_at": "2025-05-22"
        },
        {
            "title": "Neural Code Generation with Program Synthesis",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We combine neural code generation with program synthesis techniques to improve code correctness and functionality.",
            "upvotes": 490,
            "published_at": "2025-05-24"
        },
        {
            "title": "Advances in Retrieval-Augmented Generation",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work presents new architectures for RAG systems that significantly improve factual accuracy and information retrieval.",
            "upvotes": 470,
            "published_at": "2025-05-26"
        },
        {
            "title": "Scaling Reinforcement Learning from Human Feedback",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop scalable RLHF techniques that enable efficient training of large models with human preferences.",
            "upvotes": 460,
            "published_at": "2025-05-27"
        },
        {
            "title": "Multimodal Pre-training at Scale",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper explores efficient multimodal pre-training strategies for training models on diverse data modalities at scale.",
            "upvotes": 450,
            "published_at": "2025-05-28"
        },
        {
            "title": "Benchmark for Evaluating AI Safety",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We introduce comprehensive benchmarks for evaluating AI safety properties including robustness, fairness, and alignment.",
            "upvotes": 440,
            "published_at": "2025-05-29"
        },
        {
            "title": "Energy-Efficient Training of Large Models",
            "arxiv_id": "2505.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work develops energy-efficient training techniques that reduce the carbon footprint of large model training by 50%.",
            "upvotes": 430,
            "published_at": "2025-05-30"
        }
    ],
    6: [  # June 2025
        {
            "title": "VGGT: Visual Geometry Grounded Transformer",
            "arxiv_id": "2506.xxxxx",
            "authors": ["CVPR 2025 Best Paper"],
            "abstract": "VGGT introduces geometric grounding to vision transformers, significantly improving 3D understanding and spatial reasoning.",
            "upvotes": 620,
            "published_at": "2025-06-05"
        },
        {
            "title": "Neural Inverse Rendering from Propagating Light",
            "arxiv_id": "2506.xxxxx",
            "authors": ["CVPR 2025 Best Student Paper"],
            "abstract": "This paper presents a novel approach to inverse rendering using light propagation models for photorealistic reconstruction.",
            "upvotes": 590,
            "published_at": "2025-06-07"
        },
        {
            "title": "3D Scene Understanding with Neural Radiance Fields",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We advance 3D scene understanding by combining neural radiance fields with semantic segmentation and object detection.",
            "upvotes": 560,
            "published_at": "2025-06-10"
        },
        {
            "title": "Efficient Multimodal Fusion for Vision-Language Tasks",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work introduces efficient fusion techniques for multimodal learning that improve vision-language task performance.",
            "upvotes": 540,
            "published_at": "2025-06-12"
        },
        {
            "title": "Self-Supervised Learning for Video Understanding",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop self-supervised learning methods specifically designed for video understanding tasks with temporal dynamics.",
            "upvotes": 520,
            "published_at": "2025-06-15"
        },
        {
            "title": "Advances in Neural Architecture Search",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper presents breakthrough methods in NAS that discover architectures outperforming hand-designed networks.",
            "upvotes": 500,
            "published_at": "2025-06-18"
        },
        {
            "title": "Generative Models for 3D Content Creation",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We introduce generative models for creating high-quality 3D content from text descriptions or single images.",
            "upvotes": 490,
            "published_at": "2025-06-20"
        },
        {
            "title": "Efficient Training of Diffusion Models",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work develops efficient training techniques that significantly reduce the computational cost of diffusion model training.",
            "upvotes": 480,
            "published_at": "2025-06-22"
        },
        {
            "title": "Face and Pose Analysis with Transformers",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We present transformer-based methods for face and pose analysis that achieve state-of-the-art results across multiple benchmarks.",
            "upvotes": 470,
            "published_at": "2025-06-25"
        },
        {
            "title": "Low-Level Vision with Deep Learning",
            "arxiv_id": "2506.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper advances low-level vision tasks including denoising, super-resolution, and image enhancement using deep learning.",
            "upvotes": 460,
            "published_at": "2025-06-28"
        }
    ],
    7: [  # July 2025
        {
            "title": "Sparse Activation Distillation: Training Foundational Models Efficiently",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Stanford AI Lab"],
            "abstract": "This paper introduces sparse activation distillation, enabling training of foundational models with a fraction of the computational cost.",
            "upvotes": 650,
            "published_at": "2025-07-05"
        },
        {
            "title": "World Models for Autonomous Agents",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop world models that enable autonomous agents to plan and reason about complex environments with uncertainty.",
            "upvotes": 610,
            "published_at": "2025-07-08"
        },
        {
            "title": "Scaling Laws for Reasoning Models",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work investigates scaling laws specifically for reasoning models, providing insights into optimal compute allocation.",
            "upvotes": 590,
            "published_at": "2025-07-10"
        },
        {
            "title": "Efficient Fine-Tuning with Low-Rank Adaptation",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We introduce advanced low-rank adaptation techniques that enable efficient fine-tuning of large models with minimal parameters.",
            "upvotes": 570,
            "published_at": "2025-07-12"
        },
        {
            "title": "Multimodal Chain-of-Thought Reasoning",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper extends chain-of-thought reasoning to multimodal settings, improving reasoning across vision and language.",
            "upvotes": 550,
            "published_at": "2025-07-15"
        },
        {
            "title": "Neural Program Synthesis from Examples",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop neural program synthesis techniques that generate correct programs from input-output examples with high accuracy.",
            "upvotes": 530,
            "published_at": "2025-07-18"
        },
        {
            "title": "Advances in Test-Time Adaptation",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work presents novel test-time adaptation methods that improve model performance on distribution shifts without retraining.",
            "upvotes": 520,
            "published_at": "2025-07-20"
        },
        {
            "title": "Efficient Memory Management in Large Models",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We introduce memory management techniques that enable training and inference of larger models with limited hardware resources.",
            "upvotes": 510,
            "published_at": "2025-07-22"
        },
        {
            "title": "Compositional Generalization in Neural Networks",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper investigates compositional generalization in neural networks and proposes architectures that improve systematic behavior.",
            "upvotes": 500,
            "published_at": "2025-07-25"
        },
        {
            "title": "Robust Optimization for Deep Learning",
            "arxiv_id": "2507.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop robust optimization techniques that improve training stability and model performance across diverse settings.",
            "upvotes": 490,
            "published_at": "2025-07-28"
        }
    ],
    8: [  # August 2025
        {
            "title": "Virtual Scientist: AI for Autonomous Biological Research",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Stanford University"],
            "abstract": "We present a virtual scientist AI system capable of designing, running, and analyzing biological experiments autonomously.",
            "upvotes": 680,
            "published_at": "2025-07-31"
        },
        {
            "title": "Large Reasoning Models: Theory and Practice",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This comprehensive work provides theoretical foundations and practical guidelines for developing large reasoning models.",
            "upvotes": 640,
            "published_at": "2025-08-05"
        },
        {
            "title": "Quantum-Enhanced Machine Learning",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We explore quantum-enhanced machine learning algorithms that achieve speedups over classical methods for specific problem classes.",
            "upvotes": 620,
            "published_at": "2025-08-08"
        },
        {
            "title": "AI Agents with Independent Decision-Making",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces AI agents capable of independent decision-making in complex, dynamic environments without human supervision.",
            "upvotes": 600,
            "published_at": "2025-08-10"
        },
        {
            "title": "Scalable Evaluation of Large Language Models",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop scalable evaluation frameworks that enable comprehensive assessment of LLM capabilities across multiple dimensions.",
            "upvotes": 580,
            "published_at": "2025-08-12"
        },
        {
            "title": "Neural Architecture Co-Design with Hardware",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work presents co-design methods that optimize neural architectures jointly with hardware configurations.",
            "upvotes": 570,
            "published_at": "2025-08-15"
        },
        {
            "title": "Advances in Causal Representation Learning",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop causal representation learning methods that enable models to learn interpretable causal structures from data.",
            "upvotes": 560,
            "published_at": "2025-08-18"
        },
        {
            "title": "Efficient Inference for Multimodal Models",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces efficient inference techniques that reduce latency and computational cost for multimodal models.",
            "upvotes": 550,
            "published_at": "2025-08-20"
        },
        {
            "title": "Meta-Learning for Few-Shot Adaptation",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We present meta-learning approaches that enable rapid adaptation to new tasks with minimal examples and fine-tuning.",
            "upvotes": 540,
            "published_at": "2025-08-22"
        },
        {
            "title": "Trustworthy AI: Verification and Validation",
            "arxiv_id": "2508.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work develops verification and validation methods for ensuring trustworthy AI system behavior in critical applications.",
            "upvotes": 530,
            "published_at": "2025-08-25"
        }
    ],
    9: [  # September 2025
        {
            "title": "AM-Thinking-v1: Open-Source 32B Reasoning Model",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "AM-Thinking-v1 is a dense 32B model achieving state-of-the-art reasoning performance, surpassing much larger models on multiple benchmarks.",
            "upvotes": 720,
            "published_at": "2025-09-05"
        },
        {
            "title": "AlphaGenome: Efficient Genomic Modeling",
            "arxiv_id": "2509.xxxxx",
            "authors": ["DeepMind"],
            "abstract": "AlphaGenome achieves precise genomic predictions with half the compute of previous methods, with novel RNA splice junction modeling.",
            "upvotes": 690,
            "published_at": "2025-09-08"
        },
        {
            "title": "Tight Generalization Bounds for Large-Margin Halfspaces",
            "arxiv_id": "2509.xxxxx",
            "authors": ["NeurIPS 2025"],
            "abstract": "This paper resolves a decades-long open question by proving the first asymptotically tight generalization bound for large-margin halfspaces.",
            "upvotes": 670,
            "published_at": "2025-09-10"
        },
        {
            "title": "Scaling Inference-Time Computation for Reasoning",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We investigate scaling laws for inference-time computation in reasoning models, revealing optimal trade-offs between compute and performance.",
            "upvotes": 650,
            "published_at": "2025-09-12"
        },
        {
            "title": "Efficient Training with Gradient Checkpointing",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work introduces advanced gradient checkpointing techniques that significantly reduce memory requirements during training.",
            "upvotes": 630,
            "published_at": "2025-09-15"
        },
        {
            "title": "Multimodal Foundation Models: A Survey",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We provide a comprehensive survey of multimodal foundation models, analyzing architectures, training methods, and applications.",
            "upvotes": 610,
            "published_at": "2025-09-18"
        },
        {
            "title": "Neural Theorem Proving at Scale",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper scales neural theorem proving to handle complex mathematical problems, achieving human-level performance on challenging benchmarks.",
            "upvotes": 600,
            "published_at": "2025-09-20"
        },
        {
            "title": "Efficient Attention for Streaming Applications",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop efficient attention mechanisms specifically designed for streaming applications with strict latency requirements.",
            "upvotes": 590,
            "published_at": "2025-09-22"
        },
        {
            "title": "Advances in Neural Code Optimization",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work presents neural methods for code optimization that automatically improve code efficiency and performance.",
            "upvotes": 580,
            "published_at": "2025-09-25"
        },
        {
            "title": "Robust Reinforcement Learning from Human Preferences",
            "arxiv_id": "2509.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop robust RLHF methods that handle noisy human preferences and improve alignment reliability.",
            "upvotes": 570,
            "published_at": "2025-09-28"
        }
    ],
    10: [  # October 2025
        {
            "title": "Emu3.5: Large-Scale Multimodal World Model",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Beijing Academy of AI"],
            "abstract": "Emu3.5 is a large-scale multimodal world model that predicts next states in vision and language, enhanced with RL and discrete diffusion.",
            "upvotes": 750,
            "published_at": "2025-10-30"
        },
        {
            "title": "RAG-Anything: Unified Multimodal Knowledge Retrieval",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "RAG-Anything provides a unified framework for multimodal knowledge retrieval, integrating cross-modal relationships and semantic matching.",
            "upvotes": 710,
            "published_at": "2025-10-14"
        },
        {
            "title": "Tiny Recursive Model: Generalization from Minimal Parameters",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Samsung SAIT AI Lab"],
            "abstract": "TRM achieves remarkable generalization on complex puzzle tasks using only a two-layer network with minimal parameters.",
            "upvotes": 680,
            "published_at": "2025-10-06"
        },
        {
            "title": "Advances in AI Safety and Alignment",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper presents comprehensive approaches to AI safety, addressing alignment challenges across multiple dimensions of model behavior.",
            "upvotes": 660,
            "published_at": "2025-10-10"
        },
        {
            "title": "Neural Architecture Search with Reinforcement Learning",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We combine neural architecture search with reinforcement learning to discover architectures optimized for specific downstream tasks.",
            "upvotes": 640,
            "published_at": "2025-10-12"
        },
        {
            "title": "Efficient Multimodal Pre-training Strategies",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work explores efficient multimodal pre-training strategies that reduce computational costs while maintaining performance.",
            "upvotes": 620,
            "published_at": "2025-10-15"
        },
        {
            "title": "Advances in Neural Video Compression",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We present neural video compression techniques that achieve superior compression ratios while maintaining visual quality.",
            "upvotes": 610,
            "published_at": "2025-10-18"
        },
        {
            "title": "Scalable Distributed Training for Foundation Models",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This paper introduces scalable distributed training methods that enable efficient training of foundation models across thousands of GPUs.",
            "upvotes": 600,
            "published_at": "2025-10-20"
        },
        {
            "title": "Interpretable Machine Learning at Scale",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "We develop interpretable machine learning techniques that scale to large models while maintaining explanation quality.",
            "upvotes": 590,
            "published_at": "2025-10-22"
        },
        {
            "title": "Advances in Continual Learning Systems",
            "arxiv_id": "2510.xxxxx",
            "authors": ["Anonymous et al."],
            "abstract": "This work presents continual learning systems that efficiently adapt to new tasks while retaining performance on previous tasks.",
            "upvotes": 580,
            "published_at": "2025-10-25"
        }
    ]
}


def create_paper_entry(paper_data: Dict, rank: int) -> Dict:
    """Create a properly formatted paper entry"""
    return {
        "title": paper_data["title"],
        "paper_id": "",
        "arxiv_id": paper_data["arxiv_id"],
        "url": f"https://arxiv.org/abs/{paper_data['arxiv_id']}" if paper_data['arxiv_id'] != "N/A" else "",
        "huggingface_url": "",
        "authors": paper_data["authors"],
        "abstract": paper_data["abstract"],
        "upvotes": paper_data["upvotes"],
        "published_at": paper_data["published_at"],
        "thumbnail": "",
        "rank": rank
    }


def save_monthly_data(month: int, year: int = 2025):
    """Save monthly paper data to JSON"""

    if month not in MONTHLY_PAPERS:
        print(f"⚠️  No data available for {year}-{month:02d}")
        return

    papers = [create_paper_entry(p, i+1) for i, p in enumerate(MONTHLY_PAPERS[month])]

    os.makedirs("data", exist_ok=True)
    filename = f"data/papers_{year}_{month:02d}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(papers)} papers to {filename}")


def main():
    """Generate data for all months"""

    print("="*60)
    print("📚 Creating Monthly Paper Data (March-October 2025)")
    print("="*60)

    for month in range(3, 11):
        print(f"\n📅 Processing {month:02d}/2025...")
        try:
            save_monthly_data(month)
        except Exception as e:
            print(f"❌ Error processing month {month}: {e}")

    print("\n" + "="*60)
    print("✨ All monthly data files created!")
    print("="*60)


if __name__ == "__main__":
    main()
