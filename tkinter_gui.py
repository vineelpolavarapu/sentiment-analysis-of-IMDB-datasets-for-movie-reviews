import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk 
from tkinter import messagebox
import pickle
import re


model=tf.keras.models.load_model("imdb_sentiment_analysis.h5")


with open("imdb_word_index.pkl","rb") as f:
    word_index=pickle.load(f)

with open("maxlen.pkl","rb") as f:
    maxlen=pickle.load(f)


# encode user input based on IMDB word index
def encode_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    token=text.split()
    encode=[word_index.get(word,2) for word in token]
    print(f"token :{token}")
    print(f"encode :{encode}")
    return encode

# creats gui
root=tk.Tk()
root.title("Text classification with TensorFlow of IMDB Reviews")
root.geometry("1920x1080")

label=tk.Label(root,text="Enter your review",font=("Timesnewroman" ,24,"bold"))
label.pack(pady=(100,50))

label1=tk.Label(root,text="(Use maximum charecters upto 250 words)",font=("Arial",11))
label1.pack(pady=(0,20))

entry=tk.Entry(root,width=150)
entry.pack(pady=0)

def predict_review():
    review=entry.get()
    if not review.strip():
        messagebox.showwarning("Invalid input , please give the Review")
        return
    encoded_review=encode_text(review)
    padded_review=pad_sequences([encoded_review],maxlen=maxlen,padding="post",truncating="post")

    prediction=model.predict(padded_review)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negetive"
    print(f"prediction:{prediction[0][0]}")
    messagebox.showinfo("Prediction result",f"Movie review is {sentiment} ")

button=tk.Button(root,text="Prediction results",command=predict_review,width=20)
button.pack(pady=20)
root.mainloop()


