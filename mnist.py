import numpy as np
from PIL import Image
import argparse
import os

MODEL_FILE_PATH = "weights/mnist_model.npz"

x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')

# flatten and normalize
x_train = x_train.reshape(60000, 784) / 255.0

# one hot encode labels
y_onehot = np.zeros((60000, 10))
y_onehot[np.arange(60000), y_train] = 1


input_size = 784
hidden_size = 128
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A1, A2

def back_propagation(batch, y_true, learning_rate):

    global W1, b1, W2, b2
    batch_size = batch.shape[0]
    A1, A2 = forward(batch)
    
    loss = -np.mean( np.sum(y_true*np.log(A2+1e-8), axis=1))

    # dL / dZ2
    # softmax: Ak = e^z_k / sum_j(e^z_j)
    # cross-entropy loss (for one sample):
    # L = -sum_k(y_k.log(A_k))
    # derivitive: dL/dZ = A - y
    dZ2 = A2 - y_true

    # dL/dW2 = A1^T × dZ2
    dW2 = np.dot(A1.T, dZ2) / batch_size
    
    # dL/db2 = mean of dZ2
    db2 = np.mean(dZ2, axis=0)
    
    # dL/dZ1 = (dZ2 × W2^T ) × (A1 * (1 - A1))
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    
    # dL/dW1 = batch^T × dZ1
    dW1 = np.dot(batch.T, dZ1) / batch_size
    
    # ∂L/∂b1 = mean of dZ1
    db1 = np.mean(dZ1, axis=0)
    
    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    return loss

def train(batch_size, epochs, learning_rate):
    num_samples = x_train.shape[0]

    for epoch in range(epochs):

        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_onehot[indices]

        total_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            loss = back_propagation(x_batch,y_batch,learning_rate)
            total_loss += loss

            num_batches += 1

        _, predictions = forward(x_train[:1000])
        predicted_digits = np.argmax(predictions, axis=1)
        correct = np.sum(predicted_digits == y_train[:1000])
        accuracy = correct/1000
        avg_loss = total_loss / num_batches

        
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
    save_model()    
    print("Training complete.")

def save_model(file_path =MODEL_FILE_PATH):
    os.makedirs('weights', exist_ok=True)
    np.savez(file_path, W1=W1, b1=b1, W2=W2, b2=b2)
    print("Model saved")

def load_model(file_path = MODEL_FILE_PATH):
    global W1, b1, W2, b2
    if not os.path.exists(file_path):
        print("Model not found!")
        return False
    
    data = np.load(file_path)
    W1 = data['W1']
    b1 = data['b1']
    W2 = data['W2']
    b2 = data['b2']
    print(f"Model loaded from {file_path}")
    return True

def predict(x_test):
    # for .npy    
    _, A2 = forward(x_test)
    return np.argmax(A2, axis=1)

def predict_from_image(image_path):

    img = Image.open(image_path).convert('L')
    img = img.resize((28,28), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_flat = img_array.reshape(1, 28*28)

    _,A2 = forward(img_flat)
    prediction = np.argmax(A2[0])
    confidence = A2[0][prediction]

    print(f"prediction: {prediction} ({confidence*100:.1f}% confidence)")
    return prediction

def evaluate_from_tests():
    x_test = np.load('dataset/x_test.npy')
    y_test = np.load('dataset/y_test.npy')
    x_test = x_test.reshape(10000, 784) / 255.0

    predictions = predict(x_test)
    correct = np.sum(predictions==y_test)
    accuracy = correct / len(y_test)
    print(f"Test accuracy: {accuracy*100:.2f}% ({correct}/{len(y_test)})")

    print("sample predictions (first 15 tests)")
    for i in range(15):
        _, A2 = forward(x_test[i:i+1])
        pred = np.argmax(A2[0])
        confidence = A2[0][pred]
        status = "success" if pred == y_test[i] else "fail"
        print(f"Image {i:2d}: predicted {pred}, Actual {y_test[i]} ,{status} , ({confidence*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mnist neural network')
    parser.add_argument("action", help="Action u want to proceed doing on the model (train, predict)")
    parser.add_argument("--image", help="Image path that you want to predict with")

    args = parser.parse_args()
    if args.action == 'train':
        train(batch_size=32, epochs=20, learning_rate=0.1)

    elif args.action == 'predict':

        if not load_model():
            print("No trained model found, Please run 'train'")
            exit
        if args.image:
            predict_from_image(args.image)
        else:
            print("Please provice an image path using --image")
            print("Example: python3 mnist.py predict --image test/3.png")

    elif args.action == 'test':
        if not load_model():
            print("No trained model found, Please run 'train'")
            exit

        evaluate_from_tests()