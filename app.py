# import streamlit as st
# from PIL import Image
# from services.inferences import run_classification

# # -------------------------
# # Page config
# # -------------------------
# st.set_page_config(
#     page_title="Image Classification",
#     layout="wide"
# )

# st.title("🖼️ Image Classification (Python 3.13 Safe)")
# st.caption("ResNet50 & EfficientNet — Streamlit Cloud Compatible")

# # -------------------------
# # Sidebar
# # -------------------------
# st.sidebar.header("⚙️ Settings")

# model_choice = st.sidebar.radio(
#     "Select Model",
#     ["resnet", "efficientnet"],
#     format_func=lambda x: "ResNet50" if x == "resnet" else "EfficientNet-B0"
# )

# topk = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

# # -------------------------
# # Upload
# # -------------------------
# uploaded_file = st.file_uploader(
#     "Upload an image",
#     type=["jpg", "jpeg", "png"]
# )

# if not uploaded_file:
#     st.info("Upload an image to start.")
#     st.stop()

# image = Image.open(uploaded_file).convert("RGB")
# st.image(image, caption="Uploaded Image", use_column_width=True)

# # -------------------------
# # Inference
# # -------------------------
# with st.spinner("Running classification..."):
#     results = run_classification(image, model_choice, topk)

# # -------------------------
# # Results rendering
# # -------------------------
# st.markdown("---")
# st.subheader("🏷️ Classification Results")

# if not isinstance(results, list):
#     st.error("Model returned invalid output format (expected a list).")
#     st.stop()

# if len(results) == 0:
#     st.warning("No predictions returned by the model.")
#     st.stop()

# for idx, r in enumerate(results, start=1):

#     if not isinstance(r, dict):
#         st.error(f"Result #{idx} is not a dictionary: {r}")
#         continue

#     label = r.get("label")
#     confidence = r.get("confidence")

#     if label is None or confidence is None:
#         st.error(f"Result #{idx} missing required keys: {r}")
#         continue

#     try:
#         confidence = float(confidence)
#     except (TypeError, ValueError):
#         st.error(f"Invalid confidence value in result #{idx}: {confidence}")
#         continue

#     st.write(f"**{idx}. {label}** — {confidence:.3f}")

import streamlit as st
from PIL import Image
from services.inferences import run_detection
from utils.visualization import draw_boxes
from collections import Counter

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Item Object Detection",
    layout="wide"
)

st.title("🖼️ Multi-Item Object Detection")
st.caption(
    "Powered by Hugging Face hosted object detection "
    "(Streamlit Cloud · Python 3.13 safe)"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05
)

show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)

# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if not uploaded:
    st.info("Please upload an image to start object detection.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Failed to load image: {e}")
    st.stop()

st.image(image, caption="Uploaded Image", use_column_width=True)

# --------------------------------------------------
# Run detection
# --------------------------------------------------
with st.spinner("Detecting objects (this may take a moment on first run)..."):
    result = run_detection(image)

# --------------------------------------------------
# Handle errors from inference
# --------------------------------------------------
if isinstance(result, dict) and "error" in result:
    st.error(result["error"])
    st.info(
        "ℹ️ Hugging Face models may take 10–30 seconds to warm up "
        "on the first request. Please retry shortly."
    )
    st.stop()

if not isinstance(result, list):
    st.error("Unexpected detection output format.")
    st.stop()

# --------------------------------------------------
# Filter detections by confidence
# --------------------------------------------------
detections = [
    d for d in result
    if isinstance(d, dict)
    and d.get("confidence", 0) >= confidence_threshold
]

if len(detections) == 0:
    st.warning("No objects detected above the selected confidence threshold.")
    st.stop()

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.markdown("---")

if show_boxes:
    try:
        annotated = draw_boxes(image.copy(), detections)
        st.image(
            annotated,
            caption="Detected Objects",
            use_column_width=True
        )
    except Exception as e:
        st.error(f"Failed to draw bounding boxes: {e}")

# --------------------------------------------------
# Detection summary
# --------------------------------------------------
st.subheader("📦 Detected Items")

labels = [d["label"] for d in detections if "label" in d]
counts = Counter(labels)

for label, count in counts.most_common():
    st.write(f"**{label}** × {count}")

# --------------------------------------------------
# Detailed table
# --------------------------------------------------
with st.expander("🔍 Detailed detections"):
    for idx, d in enumerate(detections, start=1):
        st.write(
            f"{idx}. **{d['label']}** — "
            f"{d['confidence']:.2f} | "
            f"bbox={d['bbox']}"
        )



# import streamlit as st
# from PIL import Image
# from services.inferences import run_detection
# from utils.visualization import draw_boxes

# st.set_page_config(page_title="Multi-Item Detection", layout="wide")
# st.title("🖼️ Multi-Item Object Detection")

# uploaded = st.file_uploader("Upload image", ["jpg", "jpeg", "png"])
# if not uploaded:
#     st.stop()

# image = Image.open(uploaded).convert("RGB")
# st.image(image, caption="Uploaded Image", use_column_width=True)

# with st.spinner("Detecting objects..."):
#     detections = run_detection(image)

# annotated = draw_boxes(image.copy(), detections)

# st.markdown("---")
# st.image(annotated, caption="Detected Objects", use_column_width=True)

# st.subheader("📦 Detected Items")
# for d in detections:
#     st.write(f"**{d['label']}** — {d['confidence']:.2f}")



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
