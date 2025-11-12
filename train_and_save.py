import numpy as np
import tensorflow as tf
from pathlib import Path

SAVE_DIR = Path("model")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = SAVE_DIR / "linear_model.keras"

def train_and_save():
    rng = np.random.default_rng(42)
    X = rng.uniform(-10, 10, size=(1000, 1)).astype("float32")
    y = (3 * X + 7 + rng.normal(0, 1, size=(1000, 1))).astype("float32")

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=25, verbose=0)
    model.save(MODEL_PATH)

if __name__ == "__main__":
    train_and_save()
