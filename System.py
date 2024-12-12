import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.optimizers import Adam

# Load the dataset
import os
file_path = r"D:\Projects\recomendation engine\Dataset\flipkart_com-ecommerce_sample.csv"
df = pd.read_csv(file_path)

# Display basic info about the dataset
print("Dataset loaded successfully. Sample data:")
print(df.head())

# Fill missing values in important columns (optional, depends on the dataset)
important_columns = ['product_name', 'product_category_tree', 'description', 'brand']
df[important_columns] = df[important_columns].fillna('')

# Combine relevant features to create a single string per product for vectorization
df['combined_features'] = df[important_columns].apply(lambda x: ' '.join(x), axis=1)


# Encode product names to numerical labels
le = LabelEncoder()
df['product_label'] = le.fit_transform(df['product_name'])

# Create a train-test split
X = df['combined_features']
y = df['product_label']

# Tokenization and padding
max_words = 15000  # Increased vocabulary size for larger datasets
maxlen = 200  # Maximum length of sequences (adjust as necessary)

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=maxlen, padding='post', truncating='post')

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Build a deep learning model for recommendation with adjustments
# Build a deep learning model for recommendation with adjustments
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=maxlen),  # Embedding layer
    LSTM(128, return_sequences=True, name='lstm_1'),  # Named LSTM layer for easy access later
    Dropout(0.3),
    LSTM(64, name='lstm_2'),  # Named second LSTM layer
    Dropout(0.3),
    Dense(128, activation='relu'),  # Increased Dense layer size
    Dropout(0.3),
    Dense(len(df['product_label'].unique()), activation='softmax')  # Output layer with softmax activation
])

# Compile the model with a lower learning rate for Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Set up early stopping and train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# After training, extract embeddings for ANN
# Create a new model that outputs the LSTM layer's embeddings
embedding_model = Model(inputs=model.input, outputs=model.get_layer('lstm_2').output)  # Correctly named LSTM layer

# Get the embeddings for the entire dataset
X_embeddings = embedding_model.predict(X_padded)

# Use Approximate Nearest Neighbors (ANN) for faster similarity search
ann_model = NearestNeighbors(n_neighbors=11, algorithm='auto')

# Fit ANN model on the LSTM embeddings (not raw X_padded data)
ann_model.fit(X_embeddings)

# Function to recommend products based on the trained model and ANN
def get_recommendations(product_name):
    if product_name not in le.classes_:
        print("Product not found in the dataset.")
        return []

    # Get the product's index in the dataset
    product_idx = le.transform([product_name])[0]

    # Extract the embedding of the given product using the LSTM model
    product_embedding = embedding_model.predict(X_padded[product_idx].reshape(1, -1))

    # Use ANN to find similar products
    distances, indices = ann_model.kneighbors(product_embedding)
    similar_indices = indices.flatten()[1:]  # Get top 10 similar products excluding the product itself

    # Return recommended products
    recommended_products = df.iloc[similar_indices][['product_name', 'product_url', 'retail_price', 'discounted_price', 'brand']]

    return recommended_products

# Example usage
product_to_search = 'Get Glamr Designer Uggy Boots'  # Replace with a product name from your dataset
recommended_products = get_recommendations(product_to_search)

if not recommended_products.empty:
    print("Top 10 recommended products for", product_to_search, ":\n")
    print(recommended_products)
else:
    print("No recommendations found.")
