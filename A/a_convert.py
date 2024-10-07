import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Step 1: Load the PyTorch model from the .p file
model = torch.load('a_model.p')
model.eval()  # Set the model to evaluation mode

# Step 2: Define a dummy input for the model's input shape
# Adjust the input shape (1, 3, 224, 224) according to your model's input requirements
dummy_input = torch.randn(1, 3, 224, 224)  # Example input shape
onnx_file_path = 'model.onnx'

# Export the PyTorch model to ONNX format
torch.onnx.export(model, dummy_input, onnx_file_path,
                  input_names=['input'], output_names=['output'])

# Step 3: Convert the ONNX model to TensorFlow format
onnx_model = onnx.load(onnx_file_path)  # Load the ONNX model
tf_rep = prepare(onnx_model)  # Prepare the ONNX model for TensorFlow
tf_rep.export_graph('a_model.pb')  # Save the TensorFlow model

# Step 4: Convert the TensorFlow model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('a_model.pb')  # Load the TensorFlow model
tflite_model = converter.convert()  # Convert to TFLite model

# Save the TFLite model
with open('a_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion to TFLite completed successfully!")
