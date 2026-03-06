# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import re
from tqdm import tqdm
import csv

# === 1. Percorsi ===
IMG_FOLDER = "images"
CAPTIONS_FILE = "captions.txt"
FEATURES_FOLDER = "features"

# === 2. Estrazione feature immagini ===
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]  # shape: (2048,)

def process_all_images(img_folder, output_folder):
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    os.makedirs(output_folder, exist_ok=True)
    for fname in tqdm(os.listdir(img_folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            outfile = os.path.join(output_folder, fname.split('.')[0] + '.npy')
            if os.path.exists(outfile):
                continue
            feat = extract_features(os.path.join(img_folder, fname), model)
            np.save(outfile, feat)

# Decommenta per estrarre le feature (solo la prima volta!):
process_all_images(IMG_FOLDER, FEATURES_FOLDER)

# === 3. Preprocessing caption ===
def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9 ]+", "", caption)
    caption = "startseq " + caption.strip() + " endseq"
    return caption

def process_captions(captions_file):
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Salta l'intestazione
        for row in reader:
            if len(row) != 2:
                print(f"Riga malformata: {row}")
                continue
            img, cap = row
            img = img.strip()
            cap = "startseq " + cap.strip().lower() + " endseq"
            captions.setdefault(img, []).append(cap)
    return captions

captions = process_captions(CAPTIONS_FILE)


def create_tokenizer(captions_dict):
    all_captions = [c for caps in captions_dict.values() for c in caps]
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for caps in captions.values() for c in caps)

# === 4. Data generator ===
def data_generator(captions, features_folder, tokenizer, max_length, vocab_size, batch_size=32):
    image_files = list(captions.keys())
    X1, X2, y = [], [], []
    while True:
        np.random.shuffle(image_files)
        for img in image_files:
            feature_path = os.path.join(features_folder, img.split('.')[0] + '.npy')
            if not os.path.exists(feature_path):
                continue
            photo = np.load(feature_path)
            for cap in captions[img]:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []

# === 5. Modello ===
def create_model(vocab_size, max_length):
    # Feature extractor input (ResNet embedding)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence processor input (caption as sequence)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder (merge)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(vocab_size, max_length)
model.summary()

# === 6. Training con EarlyStopping===
batch_size = 32
steps = sum(len(c)-1 for c in captions.values()) // batch_size
train_gen = data_generator(captions, FEATURES_FOLDER, tokenizer, max_length, vocab_size, batch_size)

early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

model.fit(
    train_gen,
    epochs=50,
    steps_per_epoch=steps,
    verbose=1,
    callbacks=[early_stop]
)


# === 7. Funzione per generare caption (inference) ===
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.array([photo]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# === 8. Esempio di generazione caption (dopo il training) ===
# Carica la feature già salvata
photo = np.load('features/72964268_d532bb8ec7.npy')

# Genera la caption
caption = generate_caption(model, tokenizer, photo, max_length)
print("Caption generata:", caption)