import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Load and preprocess data
dataframe = pd.read_csv('ecg.csv', header=None)
raw_data = dataframe.values
data = raw_data[:, :-1]
labels = raw_data[:, -1]

train_data, test_data, _, _ = train_test_split(data, labels, test_size=0.2, random_state=21)
train_data = train_data.reshape(-1, 140, 1)
test_data = test_data.reshape(-1, 140, 1)

min_val = np.min(train_data)
max_val = np.max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# Noise functions
def add_gaussian_noise(data, noise_factor=0.1):
    data = data.numpy() if isinstance(data, tf.Tensor) else data
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return np.clip(data + noise, 0., 1.)

def add_salt_and_pepper_noise(data, salt_prob=0.02, pepper_prob=0.02):
    data = data.numpy() if isinstance(data, tf.Tensor) else data
    noisy = np.copy(data)
    num_salt = int(data.size * salt_prob)
    num_pepper = int(data.size * pepper_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in data.shape]
    noisy[tuple(salt_coords)] = 1
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in data.shape]
    noisy[tuple(pepper_coords)] = 0
    return noisy

def add_poisson_noise(data):
    data = data.numpy() if isinstance(data, tf.Tensor) else data
    data = np.clip(data, 0., 1.)                # remove negatives
    data = np.nan_to_num(data, nan=0.0)         # remove NaNs (if any)
    scaled = np.round(data * 255.0).astype(np.uint8)  # must be integers
    noisy = np.random.poisson(scaled).astype(np.float32) / 255.0
    return np.clip(noisy, 0., 1.)


def add_uniform_noise(data, low=-0.1, high=0.1):
    data = data.numpy() if isinstance(data, tf.Tensor) else data
    noise = np.random.uniform(low, high, data.shape)
    return np.clip(data + noise, 0., 1.)

# Generate noisy data
noisy_data_variants = {
    "Gaussian": (add_gaussian_noise(train_data), add_gaussian_noise(test_data)),
    "Salt & Pepper": (add_salt_and_pepper_noise(train_data), add_salt_and_pepper_noise(test_data)),
    "Poisson": (add_poisson_noise(train_data), add_poisson_noise(test_data)),
    "Uniform": (add_uniform_noise(train_data), add_uniform_noise(test_data)),
}

# Autoencoder
# Autoencoder using Transpose Convolution
class DenoiseAutoencoder(tf.keras.Model):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(16, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding='same')  # Output: (batch, 18, 16)
        ])

        # Decoder with Transposed Convolutions (simulated using Conv2DTranspose)
        self.decoder = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, axis=2)),  # (batch, 18, 1, 16)
            layers.Conv2DTranspose(16, (3, 1), strides=(2, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, (3, 1), strides=(2, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, (3, 1), strides=(2, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(1, (3, 1), strides=(1, 1), padding='same', activation='sigmoid'),
            layers.Lambda(lambda x: tf.squeeze(x, axis=2)),  # (batch, 144, 1)
            layers.Cropping1D(cropping=(2, 2))  # (batch, 140, 1)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Loss
def custom_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.mean_squared_error(y_true, y_pred) + \
           0.5 * tf.keras.losses.mean_absolute_error(y_true, y_pred)

# Train + Evaluate
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
results = {}
loss_histories = {}

for name, (noisy_train, noisy_test) in noisy_data_variants.items():
    print(f"\n Training on {name} noise...")
    model = DenoiseAutoencoder()
    model.compile(optimizer='adam', loss=custom_loss)
    history = model.fit(
        noisy_train, train_data,
        epochs=200,
        batch_size=256,
        validation_data=(noisy_test, test_data),
        shuffle=True,
        callbacks=[early_stop],
        verbose=0
    )
    loss_histories[name] = history.history

    print(f" Evaluating {name} denoising...")
    denoised = model.predict(noisy_test[:10])
    orig = test_data[:10].numpy().reshape(10, -1)
    den = denoised.reshape(10, -1)

    mae = mean_absolute_error(orig, den)
    mse = mean_squared_error(orig, den)
    rmse = np.sqrt(mse)
    data_range = np.max(orig) - np.min(orig)
    ssim_score = np.mean([ssim(orig[i], den[i], data_range=data_range) for i in range(len(orig))])

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f},SSIM: {ssim_score:.4f}")
    results[name] = (mae, mse, rmse, ssim_score)

# Plotting loss curves
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
for name, history in loss_histories.items():
    plt.plot(history["val_loss"], label=f"{name} Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss per Noise Type")
plt.legend()
plt.tight_layout()
plt.show()

