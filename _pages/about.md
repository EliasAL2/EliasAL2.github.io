---
permalink: /
title: "Blog Post for the Scientific Paper LISA: Reasoning Segmentation via Large Language Model by Elias Allert and Jonathan Tamm"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Introduction
======
Humans want modern technology to be as flexible as possible and an important part in that are
perception systems and their ability to execute visual recognition tasks. Those systems used to
require a user to give very specific instructions on what to identify in order be able to archieve it.
A good system needs to be able to reason and understand users' intentions based on implicit
instructions.
This blog post is about Reasoning Segmentation with focus on LISA, a large Language Instructed
Segmentation Assistant that was a groundbreaking innovation and redefined the boundaries of
this topic. LISA is so exceptional because it's able to produce accurate segmentation masks when given
a complicated description using complex reasoning and world knowledge, which was never possible
like that before.
We will first explain important fields such as Image Segmentation, Reasoning Segmentation and
Multi-Modal Large Language Models so everyone can understand this post even with little to no
prior knowledge. After that we present a benchmark for Reasoning Segmentation and all the key info for
LISA followed by an abalation and an evaluation.
We hope this blog post is valuable for fellow students and all other interested parties.

What is Image Segmantation?
======
Image Segmentation is a computer vision technique that involves partitioning an image into multiple
segments or regions to simplify its representation and make it more meaningful and easier to analyze.
The goal of segmentation is to divide an image into parts with similar attributes such as color, texture
or intensity, while ensuring that each part corresponds to a meaningful object or region with the image.
There are many different methods for performing image segmentation. LISA primarly utilizes a
technique called Deep Learning, which uses raw inputs to learn hierarchical representations of data.
These hierarchical representations enable deep learning models to learn complex patterns and
relationships within data, making it highly effective for image recognition and with that segmentation.

Why is Image Segmantation important?
======
Image Segmentation is important for object recognition and detection. Segmenting an image into
meaningful regions helps in identifying and locating objects within the image. It is already used in
many fields, to which we will come in a bit.
Segmenting an image into semantically meaningful regions also improves the understanding of its
content. By dividing the image into parts corresponding to different objects or regions, it becomes
easier to interpret and analyze the visual information present in the image.
Another significant application area is image compression. Segmentation can be used to identify
regions of interest within an image, allowing for more efficient compression and transmission of
visual data. By focusing on the most relevant parts of the image, unnecessary information can be
discarded or compressed, leading to reduced storage and bandwidth requirements.
A specific application, where image segmentation comes in place nowdays is hospitals. It is essential
for medical personnel to use image segmentation to find tumors and deseases, which you wouldn't be
able to detect otherwise.
In addition to that, autonomous vehicles wouldn't be on the market without image segmentation.
Lacking it, they wouldn't be able to detect pedestrians, obstacles or even be able to hold their lane.
These are only a few fields, where image segmentation is already in use with great importance and
there are many more.

What is a Multi-Modal Large Language Model?
======
In order to know what a Multi-Modal Large Language Model is, you need to know what a Large
Language Model is first. A Large Language Model or in short LLM is a type of artificial Intelligence
model that is capable of understanding and generating human-like text at a large scale. The most
renowned LLM is OpenAI's GPT, which is also arguably the most advanced technology
that has been published in this field yet.
So what is a Multi-Modal Large Language Model now? A Multi-Modal LLM is nothing else than an
LLM equipped with the ability to process and generate content across multiple modalities, not
just text, but also audio, images and video. In case of LISA, we provide language instructions
and it identifies the corresponding parts in the other modality: an image.

What is Reasoning Segmantation?
======
Reasoning Segmentation is the next big step in segmentation, because The mechanism is now
not only able to recognize an object and give its name (e.g., "the trash can"), it is also capable of
giving far more intricate descriptions (e.g., "something that the garbage should be put into") or
even longer sentences (e.g., ”After cooking, consuming food, and preparing for food, where can
we throw away the rest of the food and scraps?”) that can only be made with complex reasoning
or world knowledge.
LISA could only be such a breakthrough because of its exceptional ability for Reasoning
Segmentation.

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
Unfortunately is the end of this blog post, where we hopefully could give you a pretty good
understanding of everything around Reasoning Segmentation and of course the wonderful
LISA. We hope you have found this as interesting as we did when collecting all the information.
LISA is a great breakthrough in modern image segmentation and allthough there are currently
rather few everyday applications of it we are sure that this technology will have a lot of uses in
the future. We can see it being utilized in different medical fields for example assistive
technologies for individuals with visual impairments. By providing textual descriptions or
instructions, users could interact with devices to segment and understand visual scenes,
aiding in navigation, object recognition, and other tasks.
It will most likely also play a major role in future smart assistants and many other application
areas. The possibilities are pretty much endless.
