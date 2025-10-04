# Project README

This repository contains two Colab notebooks:
1. **Q1** – Vision Transformer on CIFAR-10  
2. **Q2** – Text-Driven Image & Video Segmentation  

## 📌 Q1 – Vision Transformer on CIFAR-10

### Overview
This notebook implements a **Vision Transformer (ViT)** from scratch and trains it on the **CIFAR-10 dataset**.  
The architecture includes patch embedding, multi-head self-attention, MLP layers, and stochastic depth for regularization.

### Features
- Custom implementations of:
  - Patch Embedding
  - Multi-Head Self-Attention
  - Transformer Encoder Block
  - Vision Transformer classifier
- Data augmentation with AutoAugment and normalization
- CIFAR-10 dataset loaders for training/testing
- Training loop with:
  - Warm-up learning rate scheduling (first 5 epochs)
  - Cosine Annealing scheduler
- Evaluation after each epoch

### Results
- Achieved **~84.95% accuracy** on CIFAR-10 test set after 100 epochs.

### How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision tqdm

## 📌 Q2 Zero-Shot Video Object Segmentation

This project implements a Zero-Shot Video Object Segmentation (ZVOS) pipeline using two powerful pre-trained models from the Hugging Face Transformers library: **GroundingDINO** for text-driven object detection, and the **Segment Anything Model (SAM)** for high-quality segmentation mask generation.

The process is as follows:
1.  **Text-to-Box (GroundingDINO):** The user provides a text prompt (e.g., "a dog.") to identify the target object in the **first frame** of the video. GroundingDINO returns a bounding box for the object.
2.  **Box-to-Mask (SAM):** SAM uses the bounding box from the first step as a prompt to generate a high-resolution segmentation mask for the target object in the first frame.
3.  **Mask Propagation (BBOX Tracking):** For all subsequent frames, the bounding box of the *previously generated mask* is calculated and used as the input prompt for SAM to segment the same object. This effectively tracks and segments the object across the video.

## 🚀 How to Run in Google Colab

This notebook (`q2.ipynb`) is designed to run entirely in a Google Colab environment, leveraging GPU acceleration for performance.

1.  **Open the Notebook:** Upload `q2.ipynb` to your Google Colab environment.
2.  **Set Runtime:** Go to **Runtime -> Change runtime type** and ensure that a **GPU (T4 recommended)** is selected as the hardware accelerator.
3.  **Run Cells:** Execute all code cells sequentially.
    * The first two cells install the necessary libraries (`transformers`, `torch`, `imageio`, etc.) and load the pre-trained models (`IDEA-Research/grounding-dino-base` and `facebook/sam-vit-base`).
    * The **"SINGLE IMAGE SEGMENTATION PIPELINE"** section is for a demonstration using a fixed image URL.
    * The **"VIDEO OBJECT SEGMENTATION (Final Corrected Version)"** section will prompt you to upload a video file from your local machine.

4.  **Important:** In the video segmentation cell (Section 5), you **must update** the `video_text_prompt` variable to accurately describe the object you want to segment in your uploaded video (e.g., `"a car."`, `"the person."`, etc.).
5.  **Download Output:** Once the video processing is complete, the final output file `output_segmented_video.mp4` will be saved to your Colab instance. You will need to manually download this file from the file browser (or add a cell to download it programmatically).

## ⚙️ Model Configuration (Using a standard setup)

The configuration uses the popular open-source foundational models for this zero-shot task.

| Model Component | Model/Source | Description |
| :--- | :--- | :--- |
| **Text-to-Box** | `IDEA-Research/grounding-dino-base` | GroundingDINO base model for referring expression grounding. |
| **Box-to-Mask** | `facebook/sam-vit-base` | Segment Anything Model (SAM) with a ViT-B image encoder. |
| **Device** | `cuda` (if available) / `cpu` | Processing is highly accelerated on GPU. |
| **Grounding Threshold** | `0.4` | Confidence threshold for object detection bounding boxes. |
| **Mask Propagation** | Bounding box tracking across frames. | The bounding box of the previous mask is used as the prompt for the next frame. |

## 📊 Results Table

This is an advanced qualitative task, so a quantitative accuracy metric is not typically used. Instead, the quality of the visual output (the segmented video) is the primary result.

| Overall Task | Result |
| :--- | :--- |
| **Zero-Shot Video Object Segmentation** | Successful qualitative segmentation and tracking demonstrated on uploaded video. |

## 📝 Concise Analysis (Bonus)

The combined GroundingDINO and SAM pipeline offers a powerful **zero-shot** capability, meaning the model can segment an object described in natural language without any task-specific training data. The key challenge in adapting this to video is **temporal consistency** (tracking the object). By using the bounding box of the previous frame's segmentation mask as the prompt for the current frame, the system effectively propagates the object's location and identity.

The primary limitation observed is the **temporal jitter or inaccuracy** in the propagated mask, especially in fast-moving scenes or when the target object changes shape dramatically. This is because SAM, when given only a bounding box, still selects from three potential mask outputs, and the simple bounding box propagation method does not guarantee the selection of the most temporally stable mask. More advanced video segmentation techniques (like SAM-based models with memory or optical flow) are needed to resolve this.
