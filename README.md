## Dogs vs Cats – CNN Classifier (TensorFlow/Keras)

This project trains a Convolutional Neural Network (CNN) to classify images of dogs and cats using TensorFlow/Keras. The workflow is implemented in the Jupyter notebook `DogvsCats.ipynb` and uses directory-based data pipelines with on-the-fly preprocessing.

### Project Structure
- `train/`
  - `dogs/` – ~10,000 training images
  - `cats/` – ~10,000 training images
- `test/`
  - `dogs/` – ~2,500 test images
  - `cats/` – ~2,500 test images
- `DogvsCats.ipynb` – main notebook: data loading, model, training, evaluation, predictions
- `README.md` – this file

### Tech Stack
- TensorFlow/Keras (CNN with `Sequential` API)
- Image preprocessing via `ImageDataGenerator`
- NumPy, Matplotlib/Seaborn for metrics and visualization
- scikit-learn metrics: confusion matrix, classification report, ROC/AUC

### Data Input Pipeline
- Uses `ImageDataGenerator(rescale=1./255, validation_split=0.2)` on `train/` to create:
  - `train_generator` (subset='training')
  - `validation_generator` (subset='validation')
- Typical parameters:
  - `target_size=(150, 150)`
  - `batch_size=20`
  - `class_mode='binary'`

The test set is read from `test/` with a separate `ImageDataGenerator(rescale=1./255)` and `shuffle=False` for deterministic evaluation.

### Model Architecture
`Sequential` CNN with four convolution + max-pooling blocks followed by dense layers:
- Conv2D(32, 3x3, ReLU) → MaxPool(2x2)
- Conv2D(64, 3x3, ReLU) → MaxPool(2x2)
- Conv2D(128, 3x3, ReLU) → MaxPool(2x2)
- Conv2D(128, 3x3, ReLU) → MaxPool(2x2)
- Flatten → Dense(512, ReLU) → Dense(1, Sigmoid)

Compilation and training (as in the notebook):
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`
- Example schedule: `epochs=10`, `steps_per_epoch=800`, `validation_steps=200`

### Evaluation & Visualizations
The notebook plots training/validation accuracy and loss over epochs. It also computes:
- Confusion matrix and classification report on the test set
- ROC curve and AUC

### Inference (Single Image)
Example helper for single-image prediction (matches the model's input size and scaling):

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def make_prediction(image_path, model):
    # Load and preprocess image
    img = load_img(image_path, target_size=(150, 150))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    prob = model.predict(arr)[0][0]
    label = 'dog' if prob >= 0.5 else 'cat'
    return label, float(prob)

# Example usage (adjust path):
# label, prob = make_prediction(r"C:\\Users\\anshu\\Desktop\\DogsVSCats\\some_image.jpg", model)
# print(label, prob)
```

Note: In the notebook, ensure the function uses the `image_path` parameter internally (not an undefined `img_path`).

### Requirements
You can use Anaconda or `pip`.

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

If GPU acceleration is desired, install a compatible TensorFlow build and CUDA/cuDNN per TensorFlow’s official guide.

### How to Run
1. Ensure the directory structure under `train/` and `test/` matches this repository.
2. Open `DogvsCats.ipynb` in Jupyter (Anaconda Navigator or `jupyter notebook`).
3. Run cells sequentially to:
   - Build data generators
   - Build and compile the model
   - Train (`fit`) with specified epochs/steps
   - Evaluate on test data
   - Make single-image predictions

### Common Issues
- FileNotFoundError for single-image prediction: make sure the image path exists and the prediction function references the correct variable (`image_path`).
- Keras warning about `input_shape` in `Sequential`: you can optionally use an explicit `Input(shape=(150, 150, 3))` layer as the first layer.
- Out-of-memory errors: reduce `batch_size`, image `target_size`, or steps; close other GPU/CPU-heavy apps.

### Notes
- The chosen architecture is a straightforward baseline; you can often improve accuracy with data augmentation, regularization (dropout, L2), learning-rate schedules, or transfer learning (e.g., MobileNetV2, EfficientNet) with fine-tuning.

### License
For personal/educational use. Ensure you have rights to the datasets you train on.


