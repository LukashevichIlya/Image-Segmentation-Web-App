# Image Segmentation Web Application

![](./demo.gif)

### Neural network model
The image semantic segmentation is performed via the pretrained
[Fully-Convolutional Network](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/) model from
PyTorch Hub. The backbone model applied is this app is ResNet-50.

### Application details
* For the backend service the [FastAPI](https://fastapi.tiangolo.com/) framework is applied.
* The frontend part is implemented via the [Streamlit](https://streamlit.io/) framework, which
allows faster development of web apps. Communication with the backend is performed via the
python `requests` package.

### Usage
To try out the application run the following commands on the machine with the running Docker service
```commandline
docker-compose build
docker-compose up
```
To test the UI, go to http://localhost:8501/. The FastAPI documentation will be located at 
http://localhost:8000/docs. 