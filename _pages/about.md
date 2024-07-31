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

<p align="center">
<img src="https://github.com/user-attachments/assets/6505e000-2d65-4dad-9596-771effc2324b" width="750" title=" DETR interaction">
</p>
<p align="center">
  Image taken from [7] Example of Image Segmentation
</p>

The picture above shows a standard image segmentation output. It is impressive how the model can correctly identify and segment all important areas. Unfortunately, the model only knows the name of the areas and nothing more. This is a problem because it wouldn't know which item you are referring to if you were to describe one to it. An even bigger deficiency is that you could not even describe an object to the model in the first place because it is actually impossible to interact with it.

A good model should be able to deeply understand what areas or items it is segmenting based on **world knowledge**. It should also offer **interaction possibilities** and **reasoning abilities**. Maybe even be able to hold **Multi-Turn conversations**, so it can understand users' intentions based on implicit descriptions. 
Just like this:

<p align="center">
<img src="https://github.com/user-attachments/assets/d5590401-24b4-400c-949b-25a3c9fb69aa" width="400" title="interaction examples">
</p>

This blog post is about **Reasoning Segmentation** with a special focus on a model that redefined the boundaries of
this topic. 
We will first explain important fields such as **Image Segmentation**, **Reasoning Segmentation** and
**Multi-Modal Large Language Models** so everyone can understand this post even with little
prior knowledge. Then there are two **related works** and after that, we introduce you to this special **model** and its **pipeline** followed by an **experiment**, 
some **pros and cons**, and lastly the **conclusion** and a glance at the **future** of this topic.               
We hope this blog post is valuable for fellow students and all other interested parties.


What is Image Segmentation?
======
Image Segmentation is a **computer vision technique** that involves partitioning an image into multiple
segments or regions to simplify its representation and make it more meaningful and easier to analyze.
The goal of segmentation is to divide an image into parts with similar attributes such as

- color
- texture
- intensity 

while ensuring that each part corresponds to a meaningful object or region with the image.

There are many different methods for performing image segmentation. LISA primarily utilizes a
technique called **Deep Learning**, which uses raw inputs to learn hierarchical representations of data.
These hierarchical representations enable deep learning models to learn complex patterns and
relationships within data, making it highly effective for image recognition and segmentation.

Why is Image Segmentation important?
======
Image Segmentation is important for **object recognition** and **detection**. Segmenting an image into
meaningful regions helps identify and locate objects within the image. It is already indispensable in
many fields, such as 

- image compression
- hospitals
- autonomous cars

<p align="center">
<img src="https://github.com/user-attachments/assets/72235632-7c95-4a0e-8b79-1655cbc8c19d" width="400" title="interaction examples">
</p>
<p align="center">
  Image taken from [3] Example of Image Segmentation
</p>

Segmentation can identify regions of interest within an image, allowing for more **efficient compression** and **transmission** of visual data. By focusing on the most relevant parts of the image, unnecessary information can be discarded or compressed, reducing storage and bandwidth requirements.

It is also common practice in many places for medical personnel to use image segmentation to find tumors and diseases, which you wouldn't be able to detect otherwise.

Autonomous vehicles wouldn't be able to get on the market without image segmentation. Lacking it, they wouldn't be able to detect pedestrians and obstacles or even be able to hold their lane.

These are only a few fields, where image segmentation is already in use and there are many more.



What is a Multi-Modal Large Language Model?
======
To know what a Multi-Modal Large Language Model is, you need to know what a **Large
Language Model** is first. A Large Language Model or short LLM is a type of artificial Intelligence
model that is capable of understanding and generating human-like text at a large scale. The most
renowned LLM is **OpenAI's GPT**, which is also arguably one of the most advanced technologies
that has been published in this field yet.


<p align="center">
<img src="https://github.com/user-attachments/assets/6140fe63-6eb6-48b6-ac54-0fabe5e7869e" width="400" title="interaction examples">
</p>
<p align="center">
  Image taken from [6] Example of a Multi-Modal-LLM
</p>

So what is a **Multi-Modal Large Language Model** then? A Multi-Modal LLM is nothing else than an
LLM which is equipped with the ability to process and generate content across **multiple modalities**, not
just text, but also audio, images, and video. In the case of LISA, we provide language instructions
and it identifies the corresponding parts in the other modality: an image.

What is Reasoning Segmentation?
======
Reasoning Segmentation is the next big step in segmentation because the mechanism is now
not only able to recognize an object and give its name (e.g., "the trash can"), it is also capable of
giving far more intricate descriptions (e.g., "something that the garbage should be put into") or
even longer sentences (e.g., ”After cooking, consuming food, and preparing for food, where can
we throw away the rest of the food and scraps?”) that can only be made with **complex reasoning
or world knowledge**.
LISA could only be such a breakthrough because of its exceptional ability for Reasoning
Segmentation.

Related Works
======
**1. SegNet:**

SegNet[4] is a cutting-edge deep learning architecture specifically designed for semantic image segmentation, which involves classifying each pixel in an image into a distinct category. This architecture features an encoder-decoder structure. The encoder, a streamlined version of conventional convolutional neural networks, captures essential image features through a series of convolution and pooling layers. The decoder then upsamples these features back to the original image resolution, ensuring accurate pixel classification.

A standout feature of SegNet is its use of pooling indices, which record the locations of maximum values during the pooling operations in the encoder. These indices are crucial in the decoder for precise upsampling, maintaining spatial accuracy and context. This method allows SegNet to be both memory-efficient and highly effective in tasks requiring detailed pixel-level information. 

SegNet finds applications in various fields, from autonomous driving (for road scene understanding) to medical imaging (for organ segmentation) and satellite image analysis (for land cover classification). Its ability to deliver high-resolution, accurate segmentation makes it a valuable tool in these and other areas.      
The picture below contains demonstrations of SegNet.

<p align="center">
<img src="https://github.com/user-attachments/assets/2000083b-c81a-404e-a571-10c0a98d226c" width="400" title="SegNet demonstration">
</p>
<p align="center">
  Image taken from [4]
</p>



**2. Flamingo:**

Flamingo[5] is an advanced vision-language model designed to seamlessly integrate visual and textual data. It uses a combination of pre-trained vision and language models, allowing it to excel in a variety of tasks that require understanding and generating content based on both images and text.  

The architecture of Flamingo includes a visual encoder to process images and a language model to handle text, filling the gap between these two types of data. Trained on extensive datasets containing both images and corresponding textual descriptions, Flamingo learns to understand the complex relationships between visual elements and their corresponding texts.    

This capability makes Flamingo highly effective in applications such as image captioning, where it generates descriptive captions for images, and visual question answering, where it provides accurate answers based on the content of images.      
Again, below is a little demonstration.

<p align="center">
<img src="https://github.com/user-attachments/assets/17116c91-5d9d-4452-8ef4-8c68ca9ec137" width="550" title="Flamingo demonstration">
</p>
<p align="center">
  Image taken from [5]
</p>


Introducing LISA
======
LISA is a large language-instructed segmentation assistant that **introduces reasoning to modern segmentation systems**. What sets LISA apart from other segmentation models of its kind is the fact that LISA can use **complex reasoning and real-world knowledge** to fulfill the tasks that it is given, even developing robust **zero-shot capabilities**. Not only that, but LISA is also able to output complete explanatory answers and is even capable of having entire Multi-Turn conversations with a user. This opens up huge potential for the way machines understand complex human requests.

Pipeline:
======

<p align="center">
  <img src="https://github.com/user-attachments/assets/d6f2c77f-5b85-499f-8eda-a93ec4233327" width="600" title="Pipeline of Lisa">
</p>

The Architecture of Lisa operates in a **pipeline fashion**. At one end the model is presented with an image and a (complex) textual instruction. These inputs then go through several different components in order to finally present an output image with a red segmentation mask laid over the desired object in the input image. The following explains this pipeline in more detail.


  
**Input:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/38f89f07-1818-46db-9ffb-624cffb291ab" width="350" title="Pipeline of Lisa">
</p>

The input of LISA contains only two things, an **image**, and a **query text**. The part that makes this input special is the **complexity** of the input text. With models prior to LISA, this input text could not be very complex. On the contrary. It had to be very simple and concise and explain directly what the intent behind the input is. For the image above the input would probably have to be along the lines of, "Please segment the orange in this image". 

Now with LISA, this is no longer the case. For LISAs input query one is now able to ask **long and complex questions** and even questions that do not directly reveal what the intended object is that should be segmented. LISA's inputs can now be questions like "What is the food with the most Vitamin C in this image?". You can also have longer conversations with it in which you slowly reveal what object you want to be segmented. 

What is it exactly that makes LISA capable of all these things? It's the **ability to reason and to understand and use real-world knowledge**. With these, it can understand even the most complex questions and still give accurate answers. How exactly these reasoning capabilities come to be is through the several different components of LISA.


  
**Vision Encoder:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/99fa7e0b-5aa0-464e-a1af-20f8bf4a1f64" width="400" title="Vision Encoder of Lisa">
</p>

The Vision Encoder or Vision Backbone is the first of these components. It takes the input image and extracts all of the important information out of it. It then transforms this data so it can be used in the next steps. For LISA the researchers decided to use **SAM as the Vision Backbone**. However, they also want to clarify that other similar models would have been also possible to be used here meaning this component is very **flexible**. Still, the researchers decided to use SAM. 

SAM[3] or the Segment Anything Model is an extremely powerful model for image segmentation tasks. It was trained on the largest segmentation dataset so far. 

One very important capability of SAM for the LISA model is its **zero-shot capability**. This means SAM is able to work with images it has never seen before. Obviously a very important feature for LISA since it should also be able to work with images it has never seen before and still segment the target object accurately. 

Another important aspect of SAM is that it was built with being **easily transferable** to new tasks in mind. Meaning it's very easy to incorporate SAM into other models and use its capabilities for specific tasks.


  
**Multi-Modal LLM:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2d99d50c-e937-4d1b-8c8e-23032aa6d99e" width="400" title="Multi Modal LLM of Lisa">
</p>

Next up is the Multi-Modal LLM of LISA. This one was trained using **LLaVA** [8] as a base. As an input, it takes both the image and the text and later on outputs a new text. The important part the researchers added to their MMLLM for LISA is the SEG token that was added to the vocabulary of the LLM.

This toke signifies the **request for segmentation**. So when the request for segmentation was made in the input text like in our example (With "Please output segmentation mask") the MMLLM will detect this and add a SEG token to its output. This SEG token will then later on clarify the need for a segmentation mask for the upcoming components of LISA so it makes sure that a segmentation mask is put over the correct part of the image.

The training of this MMLLM took a comparably small amount of time only taking 3 days and was relatively resource inexpensive.


  
**LoRA:**
<p align="center">
  <img src="https://github.com/user-attachments/assets/ac818b69-5e37-4f40-b492-25849f25db6f" width="400" title="LoRA of Lisa">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/e6e76570-9007-4c7b-843c-3762489f7f6a" width="300" title="LoRA of LoRA">
</p>
<p align="center">
  Image taken from [2]
</p>

A reason for this efficient and fast training is certainly LoRA[2]. LoRA or Low-Rang Adaptation is used to **perform efficient fine-tuning of language models to adapt to different kinds of tasks**. It enables effective **adjustments of the model without major changes**. It accomplishes this with two key factors. 

1. First up it **freezes the pre-trained weight** so it doesn't have to be refreshed on every single change. 

2. Secondly, it **injects trainable matrices** with low-rank structures into every layer of the model. This in turn reduces the amount of parameters that need to be trained.


  
**Decoder:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca55600e-5da1-4beb-a3f8-3a26cdab4f92" width="400" title="Decoder of Lisa">
</p>


The decoder now takes in all of the extracted visual data of the vision backbone and the embedding of the <SEG> token which clarifies the need for segmentation. With all of this information, it now constructs the **final segmentation mask** that is laid over the input image and presents this as our output.

  
**Resulting Image:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2aca212-df92-4ec9-9168-b0f430095c2d" width="300" title="Resulting Image of Lisa">
</p>


In the final image, the desired object is now **marked with a red segmentation mask**. The even more impressive part here though is not the part that is segmented but that part that isn't. Through the **high accuracy**, the entire rest of the image stays unchanged and unsegmented only highlighting the actual desired object. A feat that other models rarely achieve when given the same complex input texts that LISA was tested on.

To show that this isn't just one example here are some other segmented images with the respective input next to it.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9c0414c4-595c-4763-adf6-b532dbbe1b72" width="500" title="Resulting Image of Lisa">
</p>

  
**Training:**
Training LISA involves a meticulous approach of **data formulation** and **parameter optimization**. The training data is curated from various existing datasets. The data includes semantic segmentation datasets for multi-class labels, referring segmentation datasets for explicit object descriptions and visual question-answering datasets to maintain the model's original capabilities.

During training, LISA optimizes a weighted combination of text generation loss and segmentation mark loss. The **text generation** loss ensures the model's proficiency in generating accurate textual responses. The **segmentation mask loss** on the other hand encourages the production of high-quality segmentation results. The loss is computed using a combination of binary cross-entropy and DICE loss functions.

To fine-tune the model efficiently while at the same time preserving its learned knowledge LISA leverages techniques like **LoRA** which helps reduce the trainable parameters. Certain parts of the model are frozen like the vision backbone in order to **prevent severe forgetting**. Specific components tho like token embeddings and the decoder are fine-tuned to adapt to the segmentation task.


Experiment
======

In the following experiments, different methods and models were tested against LISA and even LISA itself was tested in **multiple different variants**. The training took place with only 8 NVIDIA 24G 3090 GPUs and with only 10,000 training steps. Additionally, the researchers had to **create their own benchmark** for the testing because, at the time of the study, there was no existing representative one.

**Reasoning Segmentation:**

In this first Reasoning Segmentation test the models were given **implicit query texts**, so the different models needed to actively reason or access world knowledge in order to fully understand the request and segment the correct objects.     
For a more exact analysis short query, long query, and overall performance were looked at separately, but no matter what LISA completely outperformed all other models, even the not finetuned 7B version of itself. The other models were just unable to truly understand the query.

<p align="center">
  <img src="https://github.com/user-attachments/assets/47ae958a-7c00-4218-b743-b094a2da1f1c" width="600" title="Reasoning Segmentation Result">
</p>

The following picture is an illustration of the Reasoning Segmentation test, showing how other models struggled to segment the right objects when given complex and implicit queries. Only LISA managed to get the right ones.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3046da4f-e9c1-467c-88b7-baa41c4a054d" width="700" title="Reasoning Segmentation Illustration">
</p>


**Vanilla Referring Segmentation:**

The Vanilla Referring Segmentation test was carried out to show that LISA is an overall capable model and even able to beat the state-of-the-art models when given **explicit queries** across different benchmarks.
Once again the researchers were able to celebrate seeing LISA prevailing against the other models in all but two of the eight categories.

<p align="center">
  <img src="https://github.com/user-attachments/assets/dfe5c229-ef38-4f7f-b0fb-80c95867dbe9" width="600" title="Vanilla Referring Segmentation Result">
</p>

**Ablation Study:**

To justify the use of certain design choices the researchers performed an **Ablation Study**.      

Firstly, they explain how while **SAM** emerged as the **preferred vision backbone** others would be also applicable in the presented framework and the choice is therefore adaptable. SAM does however **outperform the other vision-backbone models**. 

Furthermore, the Ablation study revealed that **LoRA finetuning does not yield any significant performance improvements on SAM**. It is actually inferior compared to the frozen one. (This could indicate potential limitations in fine-tuning strategies)   

**SAM Pre-trained Weight** on the other hand **significantly contributed** to the performance and enhanced it substantially.   

**Semantic segmentation datasets** played a crucial role in the training of the model and without it, performance would drop a lot. They are therefore quite important for training.   

**Data augmentation** (i.e. rephrasing text instructions) via GPT-3.5 also proved effective in boosting performance further.



Pros and Cons
======
**Pros:**
1.	Ease of Use: LISA as a model is really **easy to use**. This is mostly due to the fact that it has the capability to reason and understand and use real-world knowledge. Through this, the **user does not have to hold back with their input** and simplify it to make sure the model understands it. As mentioned LISA can understand complex and long input queries and you are even able to get an answer out of a longer conversation.
  
2.	Accurate results: LISAs results are **incredibly accurate** especially when compared to other previous models. LISA really only highlights the object that was intended by the text and does so with great accuracy.

3.  Helping further this field of research: The researchers with this study helped further the research in this field tremendously. This is not only done by providing the **powerful LISA model** but also by providing a **new extensive benchmark** against which to test future models.

**Cons:**
1.	Computational Resource Intensive: While it is less than previous approaches LISA **still requires significant computational resources**, including high-performance GPUs and specialized training infrastructure, making it less accessible for smaller research teams or organizations with limited resources.

2.	Dependency on Pre-trained Models: LISA heavily **relies on pre-trained multimodal LLMs and vision backbones**, necessitating access to large-scale pre-training datasets and computational resources for model initialization (and fine-tuning)

3.	Limited Benchmarks: Due to the novelty of this research topic the researchers had to **create their own benchmarks**. For future research, it would be better to have more benchmarks to get a more diverse array of feedback.

Future
------
Even though it might take a while till everybody has a LISA-powered robot buttler that is bringing you the TV remote we see **potential for LISA and by extension reasoning segmentation as a whole** being utilized in different medical fields for example assistive technologies for individuals with visual impairments. 

By providing textual descriptions or instructions, users could interact with devices to segment and understand visual scenes, aiding in navigation, object recognition, and other tasks. It will most likely also **play a major role in future smart assistants** and many other areas connected to robotics. The possibilities are pretty much endless.

Conclusion
------
And with that, we are at the end of this blog post. We hope you now have a good understanding of everything around reasoning segmentation and of course the wonderful **LISA model**. We hope you found this topic as interesting as we did (and still do). 

LISA marks an **amazing breakthrough** in modern image segmentation. Through its ability to reason and understand real-world knowledge LISA has huge potential for the way we interact with machines and **the way machines understand complex human requests**. It's a trailblazer in its field outperforming the competition and setting new standards.  

We also want to make a note and praise the researchers for all the effort and energy they put into this field of research not only by providing their powerful LISA model but also by providing an extensive benchmark that future researchers can use paired with all the information that was provided in this paper.  

References
---------
[1]Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, & Jiaya Jia. "Lisa: Reasoning segmentation via large language model." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.    
[2]Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen- Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models.arXiv:2106.09685, 2021.     
[3]Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv:2304.02643, 2023.    
[4]Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder- decoder architecture for image segmentation. TPAMI, 2017    
[5]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022    
[6]Shengqqiong Wu, Hao Fei, Leigang Qu, Wei Ji, & Tat-Seng Chua. "Next-gpt: Any-to-any multimodal llm." arXiv preprint arXiv:2309.05519 (2023)    
[7]Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey
Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020   
[8]Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv:2304.08485, 2023.
