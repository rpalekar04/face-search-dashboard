Face Search Dashboard

A computer vision dashboard that allows users to upload a reference face image and a video, and automatically detects whether the person in the image appears in the video. The app uses YOLOv8, MTCNN, and DeepFace for person detection and face identification.

Features

✔ Upload a reference face image
✔ Upload a video for face search
✔ Detect and track faces across video frames
✔ Compare face embeddings using cosine similarity
✔ Highlight matches in the video
✔ Display whether a match is found
✔ Easy-to-use web UI powered by Streamlit

Project Structure
face-search-dashboard/
├── app.py                # Streamlit dashboard application
├── yolov8n.pt            # YOLOv8 model weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

Installation & Setup
Prerequisites

Make sure you have the following installed on your system:

✔ Python 3.10
✔ pip

Clone the Repository
git clone https://github.com/rpalekar04/face-search-dashboard.git
cd face-search-dashboard

Install Dependencies

We recommend creating a new virtual environment:

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

Then install the requirements:

pip install -r requirements.txt

If requirements.txt is not present, then install manually:

pip install streamlit opencv-python ultralytics deepface mtcnn scipy
Download YOLO Model

The dashboard expects the yolov8n.pt file in the project root.
You can download it manually if auto-download fails:

curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
Run the App
streamlit run app.py

You will see output like:

Local URL: http://localhost:8501
Network URL: http://<your IP>:8501

Open the Local URL in your browser to use the dashboard.

How It Works

Upload Reference Image: A photo of the person you want to search for

Upload Video: Input video where people should be detected

Face Embedding Extraction: Using DeepFace (Facenet)

Frame Processing: Using YOLOv8 for person bounding boxes

Face Detection: With MTCNN inside each person region

Cosine Similarity: Match reference embedding vs detected face embeddings

Display Results: Annotated video frames + result message

Results

✔ If a match is found, the video shows a green box with “MATCH FOUND”
✔ The app also shows a success alert in the UI
✔ If no match is found, it displays a warning

Requirements

Example requirements.txt:

streamlit
opencv-python==4.8.1.78
ultralytics
deepface
mtcnn
scipy
numpy==1.26.4

Important: Use numpy < 2 to avoid compatibility issues with OpenCV and DeepFace.
