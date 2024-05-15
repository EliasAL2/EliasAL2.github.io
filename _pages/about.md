---
permalink: /
title: "Blog Post for the Scientific Paper LISA: Reasoning Segmentation via Large Language Model by Elias Allert and Jonathan Tamm"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

##Introduction##

What is Image Segmantation?
======

Why is Image Segmantation important?
======

What is a Multi-Modal Large Language Model?
======

What is Reasoning Segmantation?
======

Why Reasoning Segmantation so complex? 
======

Benchmark for Reasoning Segmantation
======



Introducing Lisa
======
**Architecture:**
The Architecture of Lisa operates in a pipeline fashion. At one end the model is 	presented with an image and a (complex) textual instruction. LISA then utilizes a 	multimodal LLM to generate a response. The important part here is the <SEG> 	token.
If the instruction includes a request for segmentation LISA proceeds to extract 	the embedding corresponding to <SEG>. This embedding is is then fed into a decode 	together with visual features that are extracted from the image by using a vision 	backbone. Now the final segmentation mask is generated highlighting the part of 	the image that was described in the instruction.

**Training:**
Training LISA invovles a meticulous apporach of data formulation and parameter 	optimization. The training data is curated from various existing datasets. The data 	includes semantic segmentation datasets for multi-class labels, referring 	segmentation datasets for explicit object descriptions and visual question answering 	datasets to maintain the model's original capabilities.
During training LISA optimizes a weghted combination of text generation loss and 	segmentation mark loss. The text generation loss ensures the models proficiency in 	generating accurate textual responsess. The segmenation mask loss on the other 	hand encourages the production of high-quality segmenation results. The loss is 	computed using a combination of binary cross-entropy and DICE loss functions.
To fine-tune the model efficiently while at the same time preserving its learned 	knowledge LISA leverages techniques like LoRA which helps reducing the trainable 	parameters (It does so by...). Certain parts of the model are frozen like the vision 	backbone in order to prevent severe forgetting. Specific componetns tho like token 	embeddings and the decoderr are fine-tuned to adapt to the segmentation task.


Experiment
======
⦁	**Network Architectuire:** LISA leveraged a multimodal LLM base, spefifically LLaVA-7B-v1-1 or LLaVA-13B-v1-1 coupled with the ViT-SAM backbone for vision processing. This architecture enables the integration of language understanding and visual perception.
⦁	**Training:** For this experiment the training was executed on high-performance hardware infrastructure, utilizing 8 NVIDIA 24G 3090 GPUs and the deepspeed engine. Furthermore AdamW optimizer was employed with carefully chosen hyperparameters ensuring stable and efficient training. The evalutation metrics include gloU and cloU with a preference for gloU due to its stability and suitability for assesing segmentation quality. gloU is the average of all per-image Intersection-over-Unions. cloU is the cumulativce intersection over the cumulative union.
⦁	**Results:** Lisa exhibited remarkable performance in reasoning segmentation tasks, surpassing existing works by achieving more than a 20% boost in gloU. For this massive succes the models proficiency in understanding implicit queries and leveraging multimodal LLMs certanly played a pivitol role.
⦁	**Vanilla referring segemnation:** To further prove the performance the reasearchers also let the model undergo a test in a vanilla referring segmenation task. Here once again LISA outperforms state-of-the-art methods across various benchmarks and is the best one in all but two of the scores.


Ablation
======
To justify the use of certain design choices the reasearchers performed an ablation study. Firstly the explain how while SAM emerged as the preferred vision backbone others would be also applicable in the presented framework and the choice is therfore adaptable. SAM does however outperform other vision-backbone models like Mask2Former-Swin-L. Still with using Mask2Former-Swin-L as the vision backbone the presented framework still outperforms previous works auch as X-Decoder.
Furthermore the Ablation study revealed that LoRA finetuning does not yield any significant performance improvments. It is actually inferior compared to the frozen one. (This could indicate potetntial limitations in fine-tuning stratetgies) SAM's pre-trained weight on the other hand significantly contributed to the performance and enhanced it substantualy.
Semantic segmentation  datasets played a crucial role in the training of the model and without it performance woul drop a lot. They are therfore quit important for training. Data augmentation (i.e rephrasing text instructions) via GPT-3.5 also proved effective in boosting performance further.


Pros and Cons
======
**Pros:**
1.	Integration of Language Understanding and Visual Perception: LISA seamlessly integrates language understanding with visual perception, enabling it to comprehend complex textual queries and produce fine-grained segmentation masks accurately.
  
2.	End-to-End Training: The end-to-end training approach of LISA allows for efficient optimization of the model's parameters, leading to improved performance and generalization across different segmentation tasks.

3.	Flexibility in Backbone Design: The model exhibits flexibility in vision backbone design choices, with the capability to utilize different backbone architectures such as ViT-H SAM and Mask2Former, providing versatility in addressing diverse segmentation challenges.

**Cons:**
1.	Computational Resource Intensive: While its less than previous approaches LISA still requires significant computational resources, including high-performance GPUs and specialized training infrastructure, making it less accessible for smaller research teams or organizations with limited resources.

2.	Dependency on Pre-trained Models: LISA heavily relies on pre-trained multimodal LLMs and vision backbones, necessitating access to large-scale pre-training datasets and computational resources for model initialization (and fine-tuning)

3.	Limited Benchmarks: Due to the novelity of this reasearche topics the reasearchers had to present their own benchmarks. This of course can lead to intrinsic bias towards the own reaserach and further independent benchmarks would be needed to authenticate the results presented. 

Conclusion and Future
------
