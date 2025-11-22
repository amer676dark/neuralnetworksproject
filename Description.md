
# EEG-Based Person Identification Using Time-Frequency CNN + Temporal RNN

## 📌 Project Overview
This project aims to identify **which person (subject 1–109)** an EEG segment belongs to using deep learning techniques.  
We use the **PhysioNet EEG Motor Movement/Imagery Dataset**, which contains EEG recordings from 109 healthy subjects performing or imagining motor tasks (left hand, right hand, both hands, both feet).

Our approach combines:
- **Time-Frequency CNN** for extracting spatial–spectral patterns from spectrograms  
- **Temporal RNN (GRU layers)** for modeling sequential/temporal dependencies across EEG windows  

This hybrid model learns individual-specific brainwave signatures that can be used for robust biometric identification.

---

## 🧠 Dataset: PhysioNet EEG Motor Movement/Imagery
- 109 subjects  
- 2 sessions recorded on different days  
- Tasks include real and imagined movements  
- High inter-session variability → ideal for testing model generalization  

We preprocess the EEG by:
- Bandpass filtering (1–40 Hz)  
- Resampling to 128 Hz  
- Segmenting into fixed-length windows  
- Computing mel-spectrograms  
- Normalizing time-frequency features  

---

## 🛠️ Project Structure

### **1. Preprocessing Notebook**
- Loads EDF EEG files  
- Selects EEG channels  
- Resamples & applies bandpass filtering  
- Segments EEG into equal windows  
- Converts each window into a mel-spectrogram  
- Produces training-ready NumPy tensors  

### **2. CNN + RNN Model Notebook**
- Builds the hybrid architecture:
  - Conv2D → BatchNorm → MaxPooling
  - Permutation + TimeDistributed Flatten
  - GRU layers for temporal processing
  - Dense softmax output for 109 subjects
- Trains the model on all spectrogram windows  
- Tracks training/validation accuracy and loss  

### **3. Performance Report**
Includes:
- Confusion matrix (reduced and full versions)
- Classification metrics (Accuracy, Precision, Recall, F1-score)
- Discussion:
  - Inter-session variability
  - Model strengths/limitations  
  - Potential improvements (multi-channel input, CSP, LSTMs, etc.)

### **4. Optional Visualizations**
- Sample spectrograms  
- t-SNE projection of learned embeddings  
- Feature distributions  

---

## 📊 Model Highlights
- **Input:** 2-second mel-spectrogram windows  
- **Feature extraction:** CNN learns frequency–time patterns  
- **Sequence modeling:** GRUs capture temporal evolution  
- **Output:** 109-class softmax classifier  
- **Evaluation:** Weighted F1 and overall accuracy  

---

## 🚀 Technologies Used
- Python (Google Colab)
- TensorFlow / Keras
- MNE-Python  
- Librosa  
- NumPy / SciPy  
- Scikit-learn  
- Matplotlib / Plotly  

---

## 👨‍🎓 Team Members

| Name              | Student ID  |
|-------------------|-------------|
| **Amer Abdelhamid** | 320230079 |
| **Farida Essam**    | 320230124 |

---

## 📁 Repository Contents

/Preprocessing/
preprocessing.ipynb
/Model/
cnn_rnn_model.ipynb
/results/
confusion_matrix.png
metrics_report.txt
tsne_visualization.png
README.md

---

## 🔧 How to Run
1. Open the notebooks in **Google Colab**  
2. Upload or link the PhysioNet dataset  
3. Run the preprocessing notebook to generate spectrograms  
4. Train the CNN+RNN model  
5. Review evaluation metrics and generated plots  

---

## 📌 Notes
- Performance depends heavily on window size, spectrogram parameters, and model depth.  
- Using **multi-channel EEG input** instead of channel averaging will significantly improve accuracy.  
- Future extensions:
  - Transformer-based temporal modeling  
  - Cross-subject domain adaptation  
  - Channel-wise attention networks  

---
