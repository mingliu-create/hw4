import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io

# Import our custom modules
from models.cnn import CNN  # We will use the CNN model for the app
from streamlit_drawable_canvas import st_canvas

# --- 1. Model Loading ---

@st.cache_resource # Cache the model loading
def load_model(model_path):
    """Loads the pre-trained PyTorch model."""
    model = CNN()
    # Load the state dict, ensuring it's mapped to the CPU
    # This is important for compatibility if the model was trained on a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# --- 2. Image Preprocessing ---

def preprocess_image(image_data):
    """
    Preprocesses the canvas image for model prediction.
    - Resizes to 28x28
    - Converts to grayscale
    - Normalizes the image
    """
    # The canvas returns a numpy array with RGBA channels
    # Convert to PIL Image to use torchvision transforms
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')

    # Convert to grayscale
    img = img.convert('L')

    # Resize to 28x28 for the model
    img = img.resize((28, 28))

    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Apply transformations and add a batch dimension (B, C, H, W)
    tensor = transform(img).unsqueeze(0)
    return tensor

# --- 3. Main App ---

st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

st.title("✏️ MNIST 手寫數字辨識")
st.write("在下方的畫布上畫一個 0 到 9 的數字，模型將會即時預測它是什麼！")

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("畫布")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with some opacity
        stroke_width=20, # Thickness of the line
        stroke_color="#FFFFFF", # Color of the line
        background_color="#000000", # Background color of the canvas
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("預測結果")
    # Placeholder for the prediction result
    prediction_text = st.empty()
    # Placeholder for the probability distribution chart
    prediction_chart = st.empty()


# --- Prediction Logic ---
if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
    # Load the trained model
    model = load_model('saved_models/cnn_mnist.pth')

    # Preprocess the drawn image
    input_tensor = preprocess_image(canvas_result.image_data)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_digit].item()

    # Display the prediction
    prediction_text.markdown(f"## 預測數字: **{predicted_digit}**")
    prediction_text.markdown(f"### (信心度: {confidence:.2f})")

    # Create a bar chart for probabilities
    prob_data = {'Digit': [str(i) for i in range(10)], 'Probability': probabilities.numpy()}
    prediction_chart.bar_chart(prob_data, x='Digit', y='Probability')
else:
    prediction_text.markdown("請在左側畫布上畫一個數字。")

st.info("專案架構：PyTorch (CNN) + Streamlit | Agent-assisted development")
