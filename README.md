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
```

### Start server :

```
uvicorn api:api --reload
```

### Make a post request :

Using Insomnia/Postman send a post request to http://127.0.0.1:8000/ with the following multipart body;

name => "image"

value => File
