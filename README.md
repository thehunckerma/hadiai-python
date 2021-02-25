# Hadi AI Face Detection API

### Conda env :

Run

```
conda create -n hadiai python=3
conda activate hadiai
conda install pillow
conda install -c conda-forge fastapi
conda install -c conda-forge uvicorn
conda install -c conda-forge opencv
conda install -c anaconda numpy
conda install -c conda-forge python-multipart
conda install -c anaconda aiofiles
conda install -c conda-forge dlib
conda install -c conda-forge face_recognition
conda install -c akode face_recognition_models
conda install -c anaconda requests
```

### Start server :

```
uvicorn api:api --reload
```

### Make a post request :

Using Insomnia/Postman send a post request to http://127.0.0.1:8000/ with the following multipart body;

name => "image"

value => File

### Face_Recognitation

Python API development for Face Recognition usig CNN and OPENCV(Cv2)
Recognize and manipulate faces from Python or from the command line with
the world’s simplest face recognition library.
Built using dlib’s state-of-the-art face recognition
