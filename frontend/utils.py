import requests

from requests_toolbelt.multipart.encoder import MultipartEncoder


def process_image(image, server_url: str) -> requests.Response:
    """Process image segmentation via post request to backend.

    Parameters
    ----------
    image
        Uploaded image
    server_url

    Returns
    -------

    """
    data = MultipartEncoder(fields={"image": ("filename", image, "image/jpg")})
    response = requests.post(server_url,
                             data=data,
                             headers={"Content-Type": data.content_type},
                             timeout=8000)
    return response
