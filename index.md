# Sudoku augmented reality

## Introduction
The main purpose of this project is to improve my skills in computer vision and machine learning. 
Sudoku in augmented reality. You can see an example on the next video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/T8cglxZA2h0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Research
All pipeline of sudoku in augmented reality can be split into several smaller processes:
<ol>
    <li>Sudoku detection process. It is a task to find sudoku on an image, or to say that there is no one.</li>
    <li>Sudoku recognition process. Based on the area that previous process consider as the sudoku deck, 
        recognize all digits inside each sell.</li>
    <li>Sudoku solver process. Based on deck that was recognized in previous steps find the solution.</li>
    <li>Sudoku drawing process. Having a solution put it back onto image.</li>
</ol>

### Detection
Detection process is one of common tasks in computer vision. For sudoku detection the next pipeline is proposed:
<ul>
    <li>Convert image into grayscale.</li>
    <li>Add contrast to image.</li>
    <li>Canny edge detection.</li>
    <li>Dilation for Canny edges.</li>
    <li>Find the biggest contour that can be approximated as polygon with 4 corners.</li>
</ul>
According to this pipeline with some thresholds (for example threshold for an area of a biggest contour) if sudoku covers
a big area of image it probably will be detected.

### Recognition
To solve this task the convolutional neural network is proposed. Using the same model architecture I achieved to get 
a good place in [kaggle mnist competition](https://www.kaggle.com/competitions/digit-recognizer/) (In 19 June 2022 i was in 172 place).
Model architecture:
```python
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
IMG_SIZE = 28
def create_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu")(inputs)
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters = 128, kernel_size = (3,3), activation="relu")(x)
    x = layers.Conv2D(filters = 128, kernel_size = (3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters = 256, kernel_size = (3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return Model(inputs, outputs, name="digit_recognizer")
```
But the handwritten digits from dataset MNIST are not suitable for sudoku recognitions. So the main part of
data were collect from [here](https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset). Also,
there were a small dataset that I collect by myself. It was based on images from common sudoku application in Ubuntu. 
My custom dataset were added, because based only on dataset from kaggle, there were occurrences that model 
confuse 1 vs 7 and vice versa.

### Solve
I turned the sudoku solving problem into problem of exact cover and implemented
the [Knuth's Algorithm X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X). As the solution is kind a bruteforce 
I decided to implement it in C++ and create bindings for Python.
