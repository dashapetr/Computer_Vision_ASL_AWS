import base64
import io
import json
import os

import boto3
import cv2
import numpy as np

from PIL import Image


sagemaker_runtime = boto3.client(
    "sagemaker-runtime"
)

ENDPOINT_NAME = os.environ[
    "SAGEMAKER_ENDPOINT"
]


# ---------------------------------------------------------
# Decode input image
# ---------------------------------------------------------

def decode_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return np.array(image)


# ---------------------------------------------------------
# Center crop to square
# ---------------------------------------------------------

def center_crop_square(image):

    height, width = image.shape[:2]

    size = min(
        height,
        width
    )

    left = (
        width - size
    ) // 2

    top = (
        height - size
    ) // 2

    return image[
        top:top + size,
        left:left + size
    ]


# ---------------------------------------------------------
# Resize to 200×200
# ---------------------------------------------------------

def resize_200(image):

    return cv2.resize(
        image,
        (200, 200),
        interpolation=cv2.INTER_AREA
    )


# ---------------------------------------------------------
# Gaussian filtering
# ---------------------------------------------------------

def gaussian_filter(image):

    return cv2.GaussianBlur(
        image,
        (5, 5),
        1.0
    )


# ---------------------------------------------------------
# Canny edges
# ---------------------------------------------------------

def canny_edges(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    return edges


# ---------------------------------------------------------
# Convert image to base64 PNG
# ---------------------------------------------------------

def image_to_base64(image):

    if len(image.shape) == 2:

        success, encoded = cv2.imencode(
            ".png",
            image
        )

    else:

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        success, encoded = cv2.imencode(
            ".png",
            bgr
        )

    if not success:

        raise RuntimeError(
            "Could not encode image"
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")


# ---------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------

def lambda_handler(event, context):

    try:

        body = event.get(
            "body",
            ""
        )

        if event.get(
            "isBase64Encoded",
            False
        ):

            image_bytes = base64.b64decode(
                body
            )

        else:

            # Frontend sends JSON containing
            # base64 image data.
            request = json.loads(body)

            image_bytes = base64.b64decode(
                request["image"]
            )

        # ---------------------------------------------
        # Preprocessing
        # ---------------------------------------------

        original = decode_image(
            image_bytes
        )

        cropped = center_crop_square(
            original
        )

        resized = resize_200(
            cropped
        )

        filtered = gaussian_filter(
            resized
        )

        edges = canny_edges(
            resized
        )

        # ---------------------------------------------
        # Send all representations to SageMaker
        # ---------------------------------------------

        payload = {
            "raw": image_to_base64(
                resized
            ),

            "filtered": image_to_base64(
                filtered
            ),

            "edges": image_to_base64(
                edges
            )
        }

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(
            response["Body"].read().decode(
                "utf-8"
            )
        )

        # ---------------------------------------------
        # Add processed images for UI
        # ---------------------------------------------

        result["images"] = {
            "raw": image_to_base64(
                resized
            ),

            "filtered": image_to_base64(
                filtered
            ),

            "edges": image_to_base64(
                edges
            )
        }

        return {
            "statusCode": 200,

            "headers": {
                "Content-Type":
                    "application/json"
            },

            "body": json.dumps(
                result
            )
        }

    except Exception as e:

        print(
            "ERROR:",
            repr(e)
        )

        return {
            "statusCode": 500,

            "headers": {
                "Content-Type":
                    "application/json"
            },

            "body": json.dumps({
                "error": str(e)
            })
        }