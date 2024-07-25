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

Introducing LISA
======
LISA is a large language instructed segementation assistant that introduces reasoning to segmenation systems. What sets LISA apart from other segmentaion models of it's kind is the fact that LISA's able to use reasoning and real world knowledge in order to understand the tasks that its given. 

Piepline:
======

<p align="center">
  <img src="https://github.com/user-attachments/assets/d6f2c77f-5b85-499f-8eda-a93ec4233327" width="600" title="Pipeline of Lisa">
</p>

The Architecture of Lisa operates in a pipeline fashion. At one end the model is 	presented with an image and a (complex) textual instruction. These inputs then go through several diffrent components in order to finally present an image with a red segmentation mask layed over the desired object in the input image. The following explains this pipeline in more detail.



**Input:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/38f89f07-1818-46db-9ffb-624cffb291ab" width="350" title="Pipeline of Lisa">
</p>

The input of LISA contains of only two things, an image and a query text. The part that makes this input special is the complexity of the input text. With models prior to LISA this input text could not be very complex. On the contrary. It actually had to be very simple and concise and explain directly what the intent behind the input is. For image above the input would probably have to be along the lines of, "please segment the orange in this image". Now with LISA this is no longer the case. For LISAs input query one is now able to ask long and complex questions and even questions that do not directly reveal what the intended object is that should be segmented. LISA's inputs can now be questions like "What is the food with the mos Vitamin C in this image?". You can also have longer conversations with it in wich you slowly reveal what object you want to be segmented. 
What is it exactly that makes Lisa capable of all these things? Its the abiltiy to reason and to understand and use real world knowledge. With these it can understand even the most complex questions and still give accurate awnsers. How exactly these reasoning capability come to be is through the several diffrent components of LISA.



**Vision Encoder:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/99fa7e0b-5aa0-464e-a1af-20f8bf4a1f64" width="400" title="Vision Encoder of Lisa">
</p>

The Vision Encoder or Vision Backbone is the first of these components. It takes the input image and extracts all of the important information out of it. It then transforms this data so it can be used in the next steps. For LISA the reaserchers decided to use SAM as the Vision Backbone. Though they also want to clarify that other similar models would have been also possible to be use here meaning this component is very flexible. Still the researchers decided to use SAM. SAM or the Segment Anything Model is an extremly powerful model for image segmentation tasks. It was trained on the largest segmentation dataset so far. One very important cababilty of SAM for the LISA model is its zero-shot capability. This means SAM is able to work with images it has never seen before. Obviously a very important feature for LISA since it should also be able to work with images it has never seen before at still segment the target object accuratly. Another important aspect of SAM is that it was build with beeing easily transferable to new tasks in mind. Meaning its very easy to incorparte SAM into other models and use its capabilitys for specific tasks.



**Multi-Modal LLM:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2d99d50c-e937-4d1b-8c8e-23032aa6d99e" width="400" title="Multi Modal LLM of Lisa">
</p>

Next up is the Multi-Modal LLM of LISA. This one was trained using LLaVa as a base. As an input it takes both the image and the text and later on outputs a new text. The important part the reasearchers added to their MMLLM for LISA is the <SEG> token that was added to the vocabulary of the LLM. This toke signifies the request for segmentation. So when the request for segmenation was made in the input text like in our example (With "Please output segmentation mask") the MMLLM will detect this and add a <SEG> token to its output. This <SEG> token will then be later on clarify the need for a segmenation mask to the upcoming components of LISA so its made sure that a segmentation mask is put over the correct part of the image.
The training of this MMLLM took a comparably small amount of time only taking 3 days and was relativly resource inexpensive.



**LoRA:**
<p align="center">
  <img src="https://github.com/user-attachments/assets/ac818b69-5e37-4f40-b492-25849f25db6f" width="400" title="LoRA of Lisa">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/e6e76570-9007-4c7b-843c-3762489f7f6a" width="300" title="LoRA of LoRA">
</p>

A reason for this efficent and fast training is certanly LoRA. LoRA or Low-Rang Adaptation is used to perform efficient fine tuning of bi language models to adapt to diffrent kinds of tasks. It enables effective adjustments of the model without major changes. It accomplishes this with two key factors. Firt up it freezes the pre trained weight so it doesn't have to be refreshed on every single change. And secondly it injects trainable matrices with low rank structures into every layer of te model. This inturn reduces the amount of parameters that need to be trained.



**Decoder:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca55600e-5da1-4beb-a3f8-3a26cdab4f92" width="400" title="Decoder of Lisa">
</p>

The decoder now takes in all of the extracted visual data of the vision backbone and the embedding of the <SEG> token wich clarifys the need for segmentation. With all of this information it now constructs the final segmentation mask that is layed over the input image and presents this as our Output.



**Resulting Image:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2aca212-df92-4ec9-9168-b0f430095c2d" width="300" title="Resulting Image of Lisa">
</p>


In the final image the desired object is now marked with a red segmentation mask. The even more impressive part here though is not the part that is segmented but that part that isn't. Through the high accuracy the entire rest of the image stays unchanged and unsgemnted only highliting the actual desired object. A feat that other models rarely achieve given the complex input texts that LISA was tested on.
To show that this isn't just a one of example here are some other segmented images with the respective input next to it.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9c0414c4-595c-4763-adf6-b532dbbe1b72" width="500" title="Resulting Image of Lisa">
</p>


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
1.	Ease of Use: LISA as a model is really easy to use. This is mostly due to the fact that it has the capability to reason and understand and use real world knowledge. Through this the user does not have to hold back with their input and simplify it to make sure the model understands it. As mentioned LISA can understand complex and long input querys and you are even able to get an awnser out of a longer conversation.
  
2.	Accurate results: LISAs results are incredibly accurate especially when compared to other previous models. LISA really only highlits that object that was intended by the text and does so with great accuracy.

3.  Helping further this field of research: The reasearchers with this study helped further the reasearche in this field tremendously. Not only by providing the powerful LISA model but also by providing a new extensive benchmark to test future models against.

**Cons:**
1.	Computational Resource Intensive: While its less than previous approaches LISA still requires significant computational resources, including high-performance GPUs and specialized training infrastructure, making it less accessible for smaller research teams or organizations with limited resources.

2.	Dependency on Pre-trained Models: LISA heavily relies on pre-trained multimodal LLMs and vision backbones, necessitating access to large-scale pre-training datasets and computational resources for model initialization (and fine-tuning)

3.	Limited Benchmarks: Due to the novelity of this reasearche topics the reasearchers had to present their own benchmarks. For future reasearche it would be better to have more benchmarks to get a more diverse array of feedback from.

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
