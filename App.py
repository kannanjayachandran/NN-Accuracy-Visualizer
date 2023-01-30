import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: red;'>Neural Network Accuracy Visualizer</h1>", unsafe_allow_html=True)

st.write('''

> This is an application that shows the accuracy of the neural network I have created to predict the handwritten digits from the MNIST dataset.

''')
# creating the dataset for finding the image
data = pd.read_csv('Data/train.csv')
data = np.array(data)
data = data.T
y_train = data[0]
x_train = data[1:]


def relu(arr):
    return np.maximum(arr, 0)


def soft_max(arr):
    sm = np.exp(arr) / sum(np.exp(arr))
    return sm


def forward_propagation(W1, W2, B1, B2, x):
    Z1 = W1.dot(x) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = soft_max(Z2)
    return Z1, Z2, A1, A2


def get_predictions(A2):
    return np.argmax(A2, 0)


def make_prediction(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(w1, w2, b1, b2, x)
    predictions = get_predictions(a2)
    return predictions


def test_predictions(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]
    predictions = make_prediction(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]

    print('prediction: ', predictions)
    print("label: ", label)

    current_image = current_image.reshape((28, 28)) * 255

    if predictions == label:
        st.success('The prediction is correct')
    else:
        st.error('The prediction is incorrect')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Prediction: ' + str(predictions))
    with col2:
        st.image(current_image, clamp=True, use_column_width=True)
    with col3:
        st.write('Label: ' + str(label))




def load_model():
    with open('NN.pickle', 'rb') as f:
        a, b, c, d = pkl.load(f)
    return a, b, c, d


w1, w2, b1, b2 = load_model()
st.write('**Enter the index of the image you want to test**')
number = st.number_input('', min_value=0, max_value=40000, value=0, step=1)

test_predictions(number, w1, b1, w2, b2)
