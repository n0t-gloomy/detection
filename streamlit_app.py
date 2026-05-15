import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
from grad_cam import create_grad_cam_visualization
import os
 
# Page configuration
st.set_page_config(
    page_title="Image Classification with Explainability",
    page_icon="🧠",
    layout="wide"
)
 
# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-bar {
        height: 30px;
        background: linear-gradient(to right, #ff6b6b, #ffd43b, #51cf66);
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
 
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
 
def main():
    st.title("🧠 Image Classification with Grad-CAM Explainability")
    st.write("Upload an image to see AI predictions with visual explanations")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model path",
        value="weights/oral.pt",
        help="Path to your YOLO .pt model file"
    )
    
    # Grad-CAM settings
    st.sidebar.subheader("Grad-CAM Settings")
    target_layer = st.sidebar.text_input(
        "Target Layer",
        value="22",
        help="Layer name to visualize (usually the last conv layer)"
    )
    
    colormap_options = {
        "Jet": 2,
        "Hot": 4,
        "Viridis": 14,
        "Parula": 12,
        "Twilight": 18
    }
    colormap_name = st.sidebar.selectbox(
        "Heatmap Colormap",
        options=list(colormap_options.keys()),
        index=0
    )
    colormap = colormap_options[colormap_name]
    
    overlay_alpha = st.sidebar.slider(
        "Heatmap Intensity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    show_grad_cam = st.sidebar.checkbox(
        "Enable Grad-CAM Visualization",
        value=True,
        help="Show which parts of the image influenced the prediction"
    )
    
    # Model input size
    model_input_size = st.sidebar.selectbox(
        "Model Input Size",
        options=[224, 320, 416, 640],
        index=0,
        help="The size your model expects as input"
    )
    
    # File uploader
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
        
        # Load model
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please check that the model path is correct and the file exists.")
            return
        
        model = load_model(model_path)
        
        if model is None:
            return
        
        # Make prediction
        with st.spinner("🔄 Processing image..."):
            try:
                # Get predictions
                results = model(image)
                result = results[0]
                
                # Extract prediction info
                if hasattr(result, 'probs') and result.probs is not None:
                    top1_name = result.names[result.probs.top1.item()]
                    top1_conf = result.probs.top1conf.item()
                    
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Display prediction
                        st.metric(
                            label="Predicted Class",
                            value=top1_name,
                            delta=None
                        )
                        
                        st.metric(
                            label="Confidence Score",
                            value=f"{top1_conf:.2%}",
                            delta=None
                        )
                        
                        # Top predictions
                        st.subheader("Top Predictions")
                        
                        top_k = min(5, len(result.names))
                        top_indices = torch.topk(result.probs.data.squeeze(), k=top_k).indices
                        
                        for idx in top_indices:
                            class_name = result.names[idx.item()]
                            confidence = result.probs.data[0, idx].item()
                            
                            col_name, col_bar = st.columns([1, 3])
                            with col_name:
                                st.write(f"**{class_name}**")
                            with col_bar:
                                st.write(f"{confidence:.2%}")
                                # Visual bar
                                st.progress(confidence, text=f"{confidence:.1%}")
                
                # Grad-CAM visualization
                if show_grad_cam:
                    st.divider()
                    st.subheader("🔍 Grad-CAM Visualization")
                    st.write(
                        "The heatmap shows which regions of the image were most important "
                        "for the model's decision. Warmer colors = higher importance."
                    )
                    
                    with st.spinner("Generating Grad-CAM..."):
                        try:
                            grad_cam_image, metadata = create_grad_cam_visualization(
                                model,
                                image,
                                target_layer=target_layer,
                                model_input_size=model_input_size
                            )
                            
                            st.image(
                                grad_cam_image,
                                caption=f"Grad-CAM for {metadata['predicted_class']} "
                                       f"(confidence: {metadata['confidence']:.2%})",
                                use_column_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error generating Grad-CAM: {e}")
                            st.info(
                                "This might be due to an incorrect layer name. "
                                "Try adjusting the 'Target Layer' setting in the sidebar."
                            )
                
                # Debug info (collapsible)
                with st.expander("🔧 Debug Information"):
                    st.write(f"**Model Architecture:** {type(model).__name__}")
                    st.write(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                    st.write(f"**Number of classes:** {len(result.names)}")
                    st.write(f"**Class names:** {list(result.names.values())}")
                    
            except Exception as e:
                st.error(f"Error during inference: {e}")
                st.error(str(e))
    
    else:
        st.info("👆 Upload an image to get started")
        
        # Example info
        with st.expander("ℹ️ How to use this app"):
            st.write("""
            1. **Upload an image** - Supported formats: JPG, JPEG, PNG
            2. **Configure settings** - Adjust model path and Grad-CAM parameters in the sidebar
            3. **View predictions** - See the model's prediction and confidence score
            4. **Understand decisions** - Use Grad-CAM to see which image regions influenced the prediction
            
            ### About Grad-CAM
            Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that:
            - Highlights important regions in the image for a specific prediction
            - Uses gradient information from the neural network
            - Creates a visual heatmap showing decision-making regions
            - Helps verify if the model is focusing on relevant features
            """)
 
if __name__ == "__main__":
    main()