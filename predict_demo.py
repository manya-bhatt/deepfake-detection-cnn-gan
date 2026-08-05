import numpy as np
from tensorflow.keras.models import load_model

def predict_image(model_path, image):
    model = load_model(model_path)

    prediction = model.predict(
        np.expand_dims(image, axis=0)
    )

    if prediction[0][0] > 0.5:
        return "Deepfake"
    else:
        return "Real"

print("Prediction module ready")
