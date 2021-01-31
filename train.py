from transformers import TFGPT2LMHeadModel, GPT2Config
from tensorflow import keras
import tensorflow as tf
import numpy as np


config = GPT2Config.from_json_file("config.json")
gpt2_model = TFGPT2LMHeadModel(config=config)

train_file = np.load("train_data-cn.npz")
input_ids = train_file["arr_0"]
print(input_ids[:2])
print(input_ids.shape)

input_act = np.array(input_ids > 0, dtype="int32")

class NaturalExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学习率自然数衰减"""
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_learning_rate * tf.math.exp(-self.decay_rate * (step / self.decay_steps))

def models():
    inp = keras.layers.Input(shape=[None], name="input_1", dtype="int32")
    act = keras.layers.Input(shape=[None], name="input_2", dtype="int32")
    logits = gpt2_model(input_ids = inp, attention_mask=act, training = True)[0]
    return keras.models.Model([inp, act], logits)

dataset = tf.data.Dataset.from_tensor_slices(({
    "input_1":input_ids[:, :-1],
    "input_2":input_act[:, :-1]
}, input_ids[:, 1:])).shuffle(1000).batch(32)

total_steps = input_ids.shape[0] // 32 * 60
print("总步数：", total_steps)
natural_exp_decay = NaturalExpDecay(initial_learning_rate=2e-5,
                                    decay_steps=total_steps,
                                    decay_rate=1e-6)

optimizer = keras.optimizers.Adam(natural_exp_decay)
model = models()
model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE), metrics=["accuracy"])
model.summary()
train_call = keras.callbacks.ModelCheckpoint("models.h5", monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=2)

model.fit(dataset, epochs=60, callbacks=[train_call])
model.save_weights("tf_model.h5")
gpt2_model.save_pretrained("gpt2-cn")
