# ECG Signal Denoising with Transpose Convolutional Autoencoder

This repository demonstrates the use of a 1D Convolutional Autoencoder with transposed convolutions to denoise ECG signals. Various types of noise are artificially added to the data, and the model learns to reconstruct the original clean signals.

---

## 📊 Dataset

- The dataset used is a CSV file (`ecg.csv`) containing raw ECG signal samples.
- Each row contains 140 time-steps followed by a label (not used for classification in this task).

---

## ⚙️ Features

- Preprocessing of ECG data for deep learning models.
- Noise addition:
  - Gaussian
  - Salt & Pepper
  - Poisson
  - Uniform
- Autoencoder built using:
  - Conv1D + MaxPooling1D for encoding
  - Conv2DTranspose for decoding (with reshaping)
- Custom loss function combining MAE and MSE.
- Performance evaluation using:
  - MAE, MSE, RMSE
  - Structural Similarity Index (SSIM)
- Visualization of loss curves across noise types.

---

## 🧠 Model Architecture

The autoencoder consists of:
- **Encoder**: Sequential stack of Conv1D layers + BatchNorm + MaxPooling.
- **Decoder**: Reshaped output followed by Conv2DTranspose layers + Cropping to match input size.

---

## 📦 Dependencies

Install the following Python libraries:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn scikit-image
```

---

## 🚀 Running the Code

1. Place `ecg.csv` in the same directory.
2. Run the main script:

```bash
python transpose.py
```

3. The script will:
   - Add different types of noise to ECG signals
   - Train a denoising autoencoder for each type
   - Evaluate the model using MAE, MSE, RMSE, and SSIM
   - Plot loss curves for comparison

---

## 📈 Results

Each noise type is evaluated with quantitative metrics and visualized using validation loss curves. SSIM and RMSE offer insights into signal reconstruction quality.

---

## 📂 Files

- `transpose.py` – Core implementation of the denoising autoencoder.
- `ecg.csv` – Raw ECG dataset.

---

## 🧑‍💻 Author

Developed by [Your Name]

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.