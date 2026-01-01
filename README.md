⚠️ Due to GitHub file size limitations, the trained autoencoder model (best_auto-encodeur.keras) is not included in this repository.

**Open and run the following notebook to get it :**
```
AutoEncoder.ipynb
```

---

## Execution Environment

All experiments were executed on **Google Colab with GPU**

---

## Inspiration

Alzheimer’s disease is a progressive neurological disorder that profoundly impacts patients, families, and healthcare systems. One of the major challenges lies in **early detection and reliable assessment of disease severity**, especially when visual differences between MRI scans can be subtle and difficult to interpret, even for trained clinicians.

We were inspired by a simple but powerful question :

**Can deep learning models not only classify Alzheimer’s stages, but also help us understand how brain structures progressively diverge from a healthy pattern?**

Rather than relying on a single modeling approach, we wanted to explore **multiple perspectives** :

*Classical supervised learning, transfer learning with state-of-the-art architectures, and finally a self-supervised anomaly detection approach that does not require labels at all.*

---

## What it does

Our project analyzes brain MRI images to study Alzheimer’s disease through **three complementary approaches** :

**1. Supervised CNN models** trained from scratch to classify MRI scans into four stages :

- Non-Demented

- Very Mild Dementia

- Mild Dementia

- Moderate Dementia

**2. Transfer Learning models** (EfficientNet and ResNet) leveraging ImageNet pretraining, fine-tuned for medical imaging.

**3. Self-supervised anomaly detection** using an **autoencoder trained only on healthy (Non-Demented) brains**, enabling :

- Detection of structural deviations

- Visualization of anomaly maps highlighting regions that diverge from the learned healthy pattern

Together, these approaches allow both **quantitative evaluation (classification performance)** and **qualitative interpretation (visual anomaly maps)** of disease progression.

---

## How we built it

**1. Supervised CNN Baselines**

We started with a simple convolutional neural network trained on grayscale MRI images (128×128).

From this baseline, we progressively improved performance by :

- Adding data augmentation

- Handling class imbalance with class weights

- Applying oversampling

- Performing hyperparameter tuning

Each model variant was evaluated independently to clearly measure incremental improvements.

**2. Transfer Learning (EfficientNet & ResNet)**

To push performance further, we implemented **transfer learning** using :

- EfficientNetB0

- ResNet50

Key design choices :

- Conversion of grayscale MRI scans to RGB

- Freezing pretrained layers initially

- Gradual **fine-tuning** of deeper layers

- Use of **adaptive learning rate scheduling** (ReduceLROnPlateau) and **early stopping**

This allowed us to benefit from rich pretrained features while adapting the models to medical imaging data.

**3. Production-like Cross-Dataset Evaluation**

To simulate real-world deployment, we tested all trained models on **previously unseen MRI datasets**.

This step revealed a critical insight :

*Models with excellent validation accuracy can collapse when faced with distribution shifts.*

This motivated the exploration of a fundamentally different paradigm.

**4. Self-Supervised Anomaly Detection with Autoencoder**

Instead of predicting labels, we trained a **convolutional autoencoder only on healthy (Non-Demented) MRI scans**.

The idea is simple :

- Learn a **compact representation of a healthy brain**

- Reconstruct healthy images accurately

- Observe **reconstruction errors** when the brain structure deviates from normality

We used :

- **Keras Tuner** to automatically search for the optimal encoder–decoder architecture

- Mean Squared Error (MSE) as reconstruction loss

- A statistically calibrated **pixel-wise anomaly threshold** based on healthy images

This enabled :

- Pixel-level anomaly maps

- Quantitative severity indicators (mean reconstruction error, anomaly ratio)

- Clear visualization of progressive structural divergence across Alzheimer’s stages

---

## Repository Structure

```
CNNs-for-Alzheimer-s-Detection/

├── AutoEncoder.ipynb
│   # Self-supervised autoencoder for anomaly detection trained on Non-Demented MRI scans

├── Kaggle_MRI_Alzheimers_Djebril_Redha_vf.ipynb
│   # Main notebook: supervised CNNs, transfer learning (EfficientNet, ResNet),
│   # production-like testing, and cross-dataset evaluation

├── OAS1_0003_MR1_mpr-3_105.jpg
├── OAS1_0004_MR1_mpr-2_116.jpg
├── OAS1_0028_MR1_mpr-2_105.jpg
├── OAS1_0308_MR1_mpr-3_123.jpg
│   # Sample brain MRI images used for qualitative testing and visualization

├── README.md
│   # Project documentation (motivation, methods, results, limitations)

├── best_model.keras
│   # Best-performing supervised CNN model (baseline / tuned)

├── best_model_v2.keras
│   # Fine-tuned CNN model trained with adaptive learning rate strategy

├── train.parquet
│   # Preprocessed training dataset metadata and labels

├── test.parquet
│   # Preprocessed test dataset metadata and labels

└── .gitignore
    # Files and folders excluded from version control
```

---

## Challenges we ran into

- **Dataset heterogeneity** : MRI scans from different sources vary in contrast, resolution, and acquisition protocols.

- **Class imbalance** : Severe stages are underrepresented, complicating supervised learning.

- **Generalization** : High validation accuracy does not guarantee robustness across datasets.

- **Interpretability** : Classification alone does not explain why a prediction is made.

- **Compute constraints** : Efficient memory handling was required to avoid GPU out-of-memory errors.

---

## Accomplishments that we are proud of

- Built a **complete experimental pipeline** from baseline CNNs to advanced self-supervised learning.

- Demonstrated the **limits of pure supervised classification** under dataset shift.

- Achieved **clear visual separation** between Alzheimer’s stages using anomaly maps.

- Designed an approach that is **label-efficient**, interpretable, and clinically intuitive.

- Combined **quantitative** metrics and **qualitative** visual explanations in a single project.

---

## What we learned

- High accuracy does not necessarily imply robustness or clinical usefulness.

- Transfer learning can significantly improve performance but still suffers from domain shift.

- Self-supervised learning is a powerful alternative when labels are scarce or unreliable.

- Autoencoders can reveal **progressive structural changes** without explicit supervision.

- Visualization is crucial for trust and interpretability in medical AI.

---

## What is next

If we had more time, we would explore :

- **Masked Autoencoders (MAE)** for stronger representation learning

- Hybrid approaches combining **anomaly scores + supervised classifiers**

- Region-based severity analysis using anatomical brain masks

---

## Disclaimer

This project was developed as part of the **AI 4 Alzheimer’s Hackathon** and is intended solely for **research, educational, and exploratory purposes**.

The models and visualizations presented in this repository are **not medical devices** and are **not intended for clinical diagnosis, treatment, or medical decision-making**.  
The highlighted anomaly regions and predictions do **not correspond to exact medical lesions**, but rather to areas where brain structures diverge from learned patterns in the data.

Any results should be interpreted with caution and **must not replace professional medical evaluation** by qualified healthcare practitioners.

---

## 👤 Authors

This project was developed by **Djebril Laouedj** and **Redha Ibbou** [@KYX6](https://github.com/KYX6), 
final-year students in **Big Data & Artificial Intelligence** at **ECE Paris**.
