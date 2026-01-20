import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from deepface import DeepFace
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Face Search in Video",
    page_icon="🎯",
    layout="wide"
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center;'> Face Search in Video</h1>
    <p style='text-align: center; color: grey;'>
    Upload a reference face image and a video to detect the person in the video
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.header("⚙️ Detection Settings")

THRESHOLD = st.sidebar.slider("Face Match Threshold", 0.3, 0.8, 0.6, 0.05)
SKIP_FRAMES = st.sidebar.slider("Process Every N Frames", 1, 10, 5)
MIN_PERSON_AREA = st.sidebar.slider("Minimum Person Area", 2000, 15000, 5000, 500)

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360

# -------------------------
# UPLOAD SECTION
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼 Upload Person Image")
    image_file = st.file_uploader(
        "Upload reference face image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if image_file:
        ref_bytes = image_file.read()
        st.session_state["ref_image"] = ref_bytes
        st.image(ref_bytes, caption="Reference Image", width="stretch")

with col2:
    st.subheader("🎬 Upload Video")
    video_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov"]
    )
    if video_file:
        st.video(video_file)

st.divider()

start = st.button("Start Face Detection", use_container_width=True)

# -------------------------
# MAIN LOGIC
# -------------------------
if image_file and video_file and start:

    st.info("🔍 Processing video... Please wait")

    match_found = False

    # Save temp files
    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img_temp.write(st.session_state["ref_image"])
    img_temp.close()

    vid_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    vid_temp.write(video_file.read())
    vid_temp.close()

    # Load models
    with st.spinner("Loading AI model..."):
        yolo = YOLO("yolov8n.pt")
        face_detector = MTCNN()
        query_embedding = DeepFace.represent(
            img_temp.name,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

    cap = cv2.VideoCapture(vid_temp.name)
    stframe = st.empty()

    frame_count = 0
    show_box_until = -1
    last_bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # Detection only on selected frames
        if frame_count % SKIP_FRAMES == 0:
            results = yolo(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if (x2 - x1) * (y2 - y1) < MIN_PERSON_AREA:
                            continue

                        person_crop = frame[y1:y2, x1:x2]
                        faces = face_detector.detect_faces(person_crop)

                        for face in faces:
                            fx, fy, fw, fh = face['box']
                            face_img = person_crop[fy:fy+fh, fx:fx+fw]

                            if face_img.size == 0:
                                continue

                            try:
                                face_embedding = DeepFace.represent(
                                    face_img,
                                    model_name="Facenet",
                                    enforce_detection=False
                                )[0]["embedding"]

                                dist = cosine(query_embedding, face_embedding)

                                if dist < THRESHOLD:
                                    match_found = True
                                    show_box_until = frame_count + 15  
                                    last_bbox = (x1, y1, x2, y2)
                            except:
                                pass

        # Draw stored box
        if show_box_until >= frame_count and last_bbox:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "MATCH FOUND",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        stframe.image(frame, channels="BGR")

    cap.release()

    if match_found:
        st.success("Match Found in the Video!")
    else:
        st.warning("No matching face found.")

    st.success("✅ Video processing completed successfully!")

elif start:
    st.warning("⚠️ Please upload both image and video before starting.")
