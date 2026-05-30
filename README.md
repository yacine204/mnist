# MNIST Neural Network (NumPy)

Simple fully‑connected neural network trained on MNIST using NumPy. Supports training, evaluating on the test set, and predicting from an image.

## Requirements

- Python 3
- NumPy
- Pillow

Install dependencies:

```bash
pip install numpy pillow
```
## Data

The dataset is expected in:

- dataset/x_train.npy
- dataset/y_train.npy
- dataset/x_test.npy
- dataset/y_test.npy

## Train

python3 mnist.py train

Sample output:

```bash
Epoch: 1/20, Loss: 0.8043, Accuracy: 0.895
Epoch: 2/20, Loss: 0.3207, Accuracy: 0.910
Epoch: 3/20, Loss: 0.2716, Accuracy: 0.924
Epoch: 4/20, Loss: 0.2386, Accuracy: 0.936
Epoch: 5/20, Loss: 0.2117, Accuracy: 0.938
Epoch: 6/20, Loss: 0.1910, Accuracy: 0.949
Epoch: 7/20, Loss: 0.1734, Accuracy: 0.955
Epoch: 8/20, Loss: 0.1591, Accuracy: 0.955
Epoch: 9/20, Loss: 0.1459, Accuracy: 0.957
Epoch: 10/20, Loss: 0.1351, Accuracy: 0.963
Epoch: 11/20, Loss: 0.1258, Accuracy: 0.964
Epoch: 12/20, Loss: 0.1174, Accuracy: 0.963
Epoch: 13/20, Loss: 0.1098, Accuracy: 0.965
Epoch: 14/20, Loss: 0.1035, Accuracy: 0.968
Epoch: 15/20, Loss: 0.0974, Accuracy: 0.968
Epoch: 16/20, Loss: 0.0921, Accuracy: 0.973
Epoch: 17/20, Loss: 0.0874, Accuracy: 0.973
Epoch: 18/20, Loss: 0.0831, Accuracy: 0.976
Epoch: 19/20, Loss: 0.0792, Accuracy: 0.973
Epoch: 20/20, Loss: 0.0753, Accuracy: 0.977
Model saved
Training complete.
```

## Test

python3 mnist.py test

Sample output:

```bash
Model loaded from weights/mnist_model.npz
Test accuracy: 97.14% (9714/10000)
sample predictions (first 15 tests)
Image  0: predicted 7, Actual 7 ,success , (99.81%)
Image  1: predicted 2, Actual 2 ,success , (99.91%)
Image  2: predicted 1, Actual 1 ,success , (99.72%)
Image  3: predicted 0, Actual 0 ,success , (99.91%)
Image  4: predicted 4, Actual 4 ,success , (99.68%)
Image  5: predicted 1, Actual 1 ,success , (99.55%)
Image  6: predicted 4, Actual 4 ,success , (99.57%)
Image  7: predicted 9, Actual 9 ,success , (99.74%)
Image  8: predicted 6, Actual 5 ,fail , (89.63%)
Image  9: predicted 9, Actual 9 ,success , (98.99%)
Image 10: predicted 0, Actual 0 ,success , (99.84%)
Image 11: predicted 6, Actual 6 ,success , (99.54%)
Image 12: predicted 9, Actual 9 ,success , (99.60%)
Image 13: predicted 0, Actual 0 ,success , (99.85%)
Image 14: predicted 1, Actual 1 ,success , (99.91%)
```
## Predict from image

![showcase](tests/custom/3.png)

```bash
python3 mnist.py predict --image tests/custom/3.png
```
Sample output:

``
Model loaded from weights/mnist_model.npz
prediction: 3 (66.6% confidence)
``
### Image tips

- Use 28×28 grayscale when possible.
- Clear foreground/background contrast improves results.
- If results are off, make sure the digit is centered and fills most of the frame.

## Model files

Trained weights are stored in:

- weights/mnist_model.npz
