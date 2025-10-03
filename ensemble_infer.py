import os, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as irv2_pre
from tensorflow.keras.applications.xception import preprocess_input as xcep_pre

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(paths, img_size=299, batch_size=16, pre_fn=None):
    def _load(p):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, 3)
        img = tf.image.resize(img, [img_size, img_size], antialias=True)
        img = tf.cast(img, tf.float32)
        return img
    ds = tf.data.Dataset.from_tensor_slices(paths).map(_load, num_parallel_calls=AUTOTUNE)
    if pre_fn is not None:
        ds = ds.map(lambda x: pre_fn(x), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def main(a):
    classes = json.load(open(a.classes))
    sample = pd.read_csv(a.sample_csv)
    sample["path"] = sample["id"].apply(lambda x: os.path.join(a.test_root, f"{x}.jpg"))
    if (~sample["path"].apply(os.path.exists)).any():
        missing = sample.loc[~sample["path"].apply(os.path.exists), "path"].head(10).tolist()
        raise FileNotFoundError(f"Missing files, e.g.: {missing[:3]}")

    m_irv2 = keras.models.load_model(a.model_irv2)
    m_iv4  = keras.models.load_model(a.model_iv4)

    ds_irv2 = build_ds(sample["path"].tolist(), img_size=a.img_size, batch_size=a.batch_size, pre_fn=irv2_pre)
    ds_iv4  = build_ds(sample["path"].tolist(), img_size=a.img_size, batch_size=a.batch_size, pre_fn=xcep_pre)

    p1 = m_irv2.predict(ds_irv2, verbose=0)
    p2 = m_iv4.predict(ds_iv4, verbose=0)

    w1, w2 = a.w_irv2, a.w_iv4
    s = (w1 + w2)
    if not np.isclose(s, 1.0):
        w1, w2 = w1 / s, w2 / s

    probs = np.clip(w1 * p1 + w2 * p2, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub = pd.DataFrame(probs, columns=classes)
    sub.insert(0, "id", sample["id"])
    sub.to_csv(a.output, index=False)
    print("Wrote:", a.output)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--sample_csv", required=True)
    ap.add_argument("--classes", default="class_index.json")
    ap.add_argument("--model_irv2", default="best_irv2.keras")
    ap.add_argument("--model_iv4", default="best_iv4.keras")
    ap.add_argument("--img_size", type=int, default=299)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--w_irv2", type=float, default=0.6)
    ap.add_argument("--w_iv4", type=float, default=0.4)
    ap.add_argument("--output", default="submission.csv")
    args = ap.parse_args()
    main(args)
