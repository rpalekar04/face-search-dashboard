# Face Search Dashboard

A powerful **Computer Vision Dashboard** that allows users to upload a reference face image and a video, and automatically detect whether the person in the image appears in the video.

Built using **YOLOv8**, **MTCNN**, **DeepFace**, and **Streamlit**, this project demonstrates real-time face search and person detection inside videos.

---

## Features

- Upload a reference face image
- Upload a video file
- Detect persons using YOLOv8
- Detect faces using MTCNN
- Extract embeddings using DeepFace (Facenet)
- Compare faces using cosine similarity
- Highlight matched person with bounding box
- Adjustable detection settings
- Clean and interactive Streamlit UI

---

## Project Structure

```
face-search-dashboard/
│
├── app.py                 # Main Streamlit application
├── yolov8n.pt             # YOLOv8 model weights
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
```

---

## Installation & Setup

### Prerequisites

Make sure you have:

- Python 3.10 (Recommended)
- pip

Python 3.12 may cause compatibility issues with NumPy and OpenCV.

---

## Clone the Repository

```bash
git clone https://github.com/rpalekar04/face-search-dashboard.git
cd face-search-dashboard
```

---

## Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

If not, install manually:

```bash
pip install streamlit opencv-python ultralytics deepface mtcnn scipy numpy==1.26.4
```

---

## Important: NumPy Compatibility Fix

If you encounter errors like:

```
ImportError: numpy.core.multiarray failed to import
AttributeError: _ARRAY_API not found
```

Run:

```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

---

## Run the Application

```bash
streamlit run app.py
```

After running, you will see:

```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open the **Local URL** in your browser.

---

## How It Works

1. User uploads a reference image.
2. DeepFace extracts face embedding from the image.
3. User uploads a video.
4. Each video frame is processed.
5. YOLOv8 detects persons in the frame.
6. MTCNN detects faces inside each detected person box.
7. DeepFace extracts embeddings from detected faces.
8. Cosine similarity compares embeddings.
9. If similarity is below threshold → MATCH FOUND.
10. Bounding box appears on matched person.

---

## Output Behavior

### If Match Found
- Green bounding box around person
- “MATCH FOUND” label on video
- Success message displayed

### If No Match Found
- Warning message displayed
- No bounding box drawn

---

## Example requirements.txt

```txt
streamlit
opencv-python==4.8.1.78
ultralytics
deepface
mtcnn
scipy
numpy==1.26.4
```

---

## Technologies Used

- Streamlit
- YOLOv8 (Ultralytics)
- DeepFace (Facenet Model)
- MTCNN
- OpenCV
- NumPy
- SciPy

---

## Future Improvements

- Add timestamp when match is found
- Save processed video with annotations
- Add confidence score display
- Add face tracking
- Add real-time webcam detection
- Convert to FastAPI + React frontend
- Deploy on AWS / GCP / Azure

---




