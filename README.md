# AIRL ASSIGNMENT

This repository contains two Colab notebooks:
1. **Q1** – Vision Transformer on CIFAR-10  
2. **Q2** – Text-Driven Image & Video Segmentation  

# 🧠 Vision Transformer (ViT) for CIFAR-10 Classification

This project implements a **Vision Transformer (ViT)** from scratch using **PyTorch** and trains it on the **CIFAR-10** dataset to classify 10 image classes.  
The implementation follows the core ViT architecture:
- Image patchification  
- Learnable positional embeddings  
- A prepended `[CLS]` token for classification  
- Stacked Transformer Encoder blocks (Multi-Head Self-Attention + MLP with residual connections and layer normalization)

---

## 🚀 How to Run in Google Colab

1. **Open Google Colab**  
   Navigate to [Google Colab](https://colab.research.google.com) and create a new **Python 3** notebook.

2. **Upload the Code**  
   Copy the entire Python script (`q1.ipynb` content) into a single code cell in your Colab notebook.

3. **Enable GPU**  
   Go to **Runtime → Change runtime type** → Select **GPU** (e.g., T4 or P100).

4. **Install/Import Dependencies**  
   The code imports all required libraries:  
   `torch`, `torchvision`, `tqdm`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.  
   Run the cell to install/import all dependencies and define the model architecture.

5. **Run the Training**  
   Execute the main code cell. The `if __name__ == '__main__':` block will:
   - Initialize the model and data loaders (downloads CIFAR-10 if needed)  
   - Train for **150 epochs** with **early stopping (patience = 20)**  
   - Automatically save the best model checkpoint as **`best_vit_cifar10_checkpoint.pth`**  
   - Print final accuracy and display the **Confusion Matrix**

---

## ⚙️ Best Model Configuration (CFG Class)

| Parameter | Value | Description |
|------------|--------|-------------|
| `img_size` | 32 | CIFAR-10 image size (32×32) |
| `patch_size` | 4 | Patch size (4×4) → 8×8 = 64 patches |
| `embed_dim` | 512 | Embedding dimension |
| `depth` | 6 | Number of Transformer Encoder blocks |
| `num_heads` | 8 | Attention heads in MHSA |
| `mlp_ratio` | 4.0 | MLP hidden layer expansion (512×4 = 2048) |
| `batch_size` | 256 | Batch size |
| `epochs` | 150 | Max training epochs |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `weight_decay` | 0.05 | Weight decay for regularization |
| **Loss** | CrossEntropyLoss (Label Smoothing = 0.1) | Classification loss |
| **Augmentations** | TrivialAugmentWide, RandomHorizontalFlip, RandomErasing (p=0.1) | Data augmentation |
| **Regularization** | MixUp (α=0.8), DropPath (0.1), Dropout (0.1) | Regularization methods |

---

## 📈 Results

The Vision Transformer was trained for up to **150 epochs** on CIFAR-10 with the configuration above.

| Metric | Value |
|---------|--------|
| **Overall Test Accuracy** | **85.82%** |
| **Total Trainable Parameters** | **18,979,338** |

---

## 💡 Concise Analysis (Bonus)

The final accuracy of **85.82%** for a moderate-sized ViT-Base model (≈19M parameters) demonstrates how well ViTs can perform even on small datasets like CIFAR-10 when combined with strong regularization and optimization.

### Key Insights

- **Patch Size (4×4)**  
  Small patch sizes (→ 64 patches) retain fine-grained spatial details — critical for CIFAR-10’s 32×32 images where convolutional inductive biases are absent.

- **Regularization (MixUp / Label Smoothing / DropPath)**  
  ViTs tend to overfit small datasets. Using these three together:
  - **MixUp (α=0.8)** blends images and labels  
  - **Label Smoothing (0.1)** reduces overconfidence  
  - **DropPath (0.1)** encourages redundant representations across layers  

  → This synergy was essential for better generalization.

- **Optimization**  
  - **AdamW** optimizer for decoupled weight decay  
  - **Cosine Annealing LR Scheduler** for smooth decay  
  - **PyTorch AMP (FP16)** for faster mixed-precision training on Colab GPU  

Together, these techniques yield robust convergence and strong test-time generalization.

---

## 🏁 Checkpoints and Outputs
-  `best_vit_cifar10_checkpoint.pth` → Best model weights  
-   Confusion Matrix → Displayed at the end of training  
-   Training Logs → Printed via `tqdm` progress bars  
-   Accuracy curves → Optional Matplotlib visualization  

---  
> **Dataset:** CIFAR-10 (Krizhevsky, 2009)  
> **Framework:** PyTorch  
> **License:** MIT License



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
