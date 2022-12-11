import io

from fastapi import FastAPI, File
from starlette.responses import Response

from segmentation_utils import get_segmentation_map, get_segmentation_model

model = get_segmentation_model()

app = FastAPI(title="Image segmentation",
              description="Get segmentation map via FCN model from PyTorch Hub.",
              version="0.1.0")


@app.post("/segmentation")
def get_segmentation(image: bytes = File(...)) -> Response:
    """Provide response with image segmentation map.

    Parameters
    ----------
    image
        Image to process.

    Returns
    -------
        Response with segmentation map of the image.

    """
    segmentation_map = get_segmentation_map(model, image)
    io_bytes = io.BytesIO()
    segmentation_map.save(io_bytes, format="PNG")
    return Response(io_bytes.getvalue(), media_type="image/png")
