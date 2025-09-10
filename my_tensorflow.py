import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers , models
import pickle

#load IMDB datasets with top most frequent 15000 words.
(train_data , train_labels) , (test_data , test_labels) = imdb.load_data(num_words=15000)

#define maximum review length beacuse sequence will be padded/trucated within the length.
maxlen=250

#padding or truncating train/test data sequence to fixed length.
#padding means add 0's to the end of the sequence as well as truncating is cutting the edge of the sequence within the fixed length.

train_data=pad_sequences(train_data,maxlen=maxlen,padding="post",truncating="post")
test_data=pad_sequences(test_data,maxlen=maxlen,padding="post",truncating="post")


vocabulary_size= 15000
embedding_dimension=32     #higher embedding dimensions for richer word representations.


# Build the model architechture
model=models.Sequential([
    layers.Embedding(vocabulary_size,embedding_dimension,input_length=250),   # Converts the word indices into embeddings such as numbers.
    layers.Bidirectional(layers.LSTM(60)),     # Keeps the word order.
    layers.Dense(32,activation="relu"),      # Lear complex patterns from review representation.
    layers.Dense(1,activation="sigmoid")     # Output layer for binary classifications which is Positive/Negetive.

])

model.summary()

# compile the model with optimizer,Loss,metrics.
model.compile(
    optimizer="adam",                   # adam algorithm is efficent for updating weights.
    loss="binary_crossentropy",         # loss function for binary classification.
    metrics=["accuracy"]                # to Track accuracy during the training.
)

# Train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
training_model=model.fit(
    train_data,train_labels,
    epochs=20,                          # No.of times model will go through the training data. 
    batch_size=512,                     # No.of samples processed before the update weights during training.
    validation_split=0.2,               # 20% of training data will set to be for validation during the process.
    verbose=1,                          # controls the progress display. 1 shows a progress bar for each epoch.
    callbacks=[early_stop]
)

# evalute the model
evalute_model=model.evaluate(test_data,test_labels,verbose=2)         # verbose 2 for show one line for each epoch.
print(f"test loss:{evalute_model[0]} , test accuracy:{evalute_model[1]}")         #for floating decimal points


# save the model 
model.save("imdb_sentiment_analysis.h5")

# load IMDB word index
word_index=tf.keras.datasets.imdb.get_word_index()
word_index={k:(v+3) for k , v in word_index.items()}

word_index["<PAD>"]=0                   # 0 for padding
word_index["<START>"]=1                 # 1 for starting
word_index["<UNK>"]=2                   # 2 for unknown words
word_index["<UNUSED>"]=3                # 3 for unused places

with open("imdb_word_index.pkl","wb") as f:
    pickle.dump(word_index,f)

with open("maxlen.pkl","wb") as f:
    pickle.dump(maxlen,f)


