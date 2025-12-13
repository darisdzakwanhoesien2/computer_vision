import streamlit as st
from PIL import Image
from services.inferences import run_classification

st.set_page_config(
    page_title="Image Classification",
    layout="wide"
)

st.title("🖼️ Image Classification (Python 3.13 Safe)")
st.caption("ResNet50 & EfficientNet — Streamlit Cloud Compatible")

# Sidebar
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.radio(
    "Select Model",
    ["resnet", "efficientnet"],
    format_func=lambda x: "ResNet50" if x == "resnet" else "EfficientNet-B0"
)

topk = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

# Upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if not uploaded_file:
    st.info("Upload an image to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# Inference
results = run_classification(image, model_choice, topk)

st.markdown("---")
st.subheader("🏷️ Classification Results")

for r in results:
    st.write(f"**{r['label']}** — {r['confidence']:.3f}")


# import streamlit as st
# from PIL import Image
# from services.inferences import run_classification

# st.set_page_config(
#     page_title="Image Classification",
#     layout="wide"
# )

# st.title("🖼️ Image Classification (Python 3.13 Safe)")
# st.caption("ResNet50 & EfficientNet — Streamlit Cloud Compatible")

# # Sidebar
# st.sidebar.header("⚙️ Settings")

# model_choice = st.sidebar.radio(
#     "Select Model",
#     ["resnet", "efficientnet"],
#     format_func=lambda x: "ResNet50" if x == "resnet" else "EfficientNet-B0"
# )

# topk = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

# # Upload
# uploaded_file = st.file_uploader(
#     "Upload an image",
#     type=["jpg", "jpeg", "png"]
# )

# if not uploaded_file:
#     st.info("Upload an image to start.")
#     st.stop()

# image = Image.open(uploaded_file).convert("RGB")
# st.image(image, caption="Uploaded Image", use_column_width=True)

# # Inference
# results = run_classification(image, model_choice, topk)

# st.markdown("---")
# st.subheader("🏷️ Classification Results")

# for r in results:
#     st.write(f"**{r['label']}** — {r['confidence']:.3f}")

# import streamlit as st
# from PIL import Image
# from services.inference import run_inference
# from services.postprocess import aggregate_classes

# st.set_page_config(
#     page_title="Multi-Item Image Classification",
#     layout="wide"
# )

# st.title("🖼️ Multi-Item Image Classification")
# st.caption("YOLOv8n (Fast) with Faster R-CNN (Research Backup)")

# # -------------------------
# # Sidebar
# # -------------------------
# st.sidebar.header("⚙️ Settings")

# model_choice = st.sidebar.radio(
#     "Select Model",
#     ["yolo", "faster_rcnn"],
#     format_func=lambda x: "YOLOv8n (Fast)" if x == "yolo" else "Faster R-CNN (Accurate)"
# )

# confidence = st.sidebar.slider(
#     "Confidence Threshold",
#     0.1, 0.9, 0.3, 0.05
# )

# # -------------------------
# # Upload
# # -------------------------
# uploaded_file = st.file_uploader(
#     "Upload an image",
#     type=["jpg", "jpeg", "png"]
# )

# if not uploaded_file:
#     st.info("Please upload an image to start.")
#     st.stop()

# image = Image.open(uploaded_file).convert("RGB")
# st.image(image, caption="Uploaded Image", use_column_width=True)

# # -------------------------
# # Inference
# # -------------------------
# classes, annotated = run_inference(
#     image=image,
#     model_name=model_choice,
#     confidence=confidence
# )

# counts = aggregate_classes(classes)

# # -------------------------
# # Display
# # -------------------------
# st.markdown("---")
# col1, col2 = st.columns(2)

# with col1:
#     st.image(annotated, caption="Detected Objects", use_column_width=True)

# with col2:
#     st.subheader("🏷️ Classification Result")
#     if counts:
#         for cls, cnt in counts.items():
#             st.write(f"**{cls}** × {cnt}")
#     else:
#         st.warning("No objects detected.")
