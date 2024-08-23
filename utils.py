import cv2
import numpy as np
import tifffile
from io import BytesIO
import base64
from PIL import Image


def read_tiff(file_path, channel=0):
    """
    Reads a TIFF file and returns the data as a numpy array.
    """

    tif = tifffile.TiffFile(file_path)
    tif_data = tif.asarray()[channel]

    return tif_data

def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


def get_channel_image(tiff_data, channel):
    """
    Returns the image for the specified channel as a base64 encoded string.
    """
    # Get the channel data
    channel_data = tiff_data[:, :, channel]
    
    # Normalize the data to 0-255 range
    channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data)) * 255
    
    # Convert the data to uint8 type
    channel_data = channel_data.astype(np.uint8)
    
    # Convert the data to a base64 encoded string
    return tifffile.imwrite(channel_data)

def numpy_to_b64_string(image):
    rawBytes = BytesIO()
    im = Image.fromarray(image)
    im.save(rawBytes, format="JPEG")
    rawBytes.seek(0)
    image = base64.b64encode(rawBytes.getvalue())
    img_str = image.decode('utf-8')
    return img_str

def extract_overlay(image_path, vmin_bf_channel=0, vmax_bf_channel=40000, vmin_overlay_red_channel=4, vmax_overlay_red_channel=400, path=True):
    if path is True:
        tiff = tifffile.TiffFile(image_path)
    else:
        tiff = image_path
    red = cv2.cvtColor(map_uint16_to_uint8(tiff.asarray()[1], lower_bound=vmin_overlay_red_channel, upper_bound=vmax_overlay_red_channel), cv2.COLOR_GRAY2BGR)
    red[:,:,2]=0
    red[:,:,1]=0
    gray = cv2.cvtColor(map_uint16_to_uint8(tiff.asarray()[0], lower_bound=vmin_bf_channel, upper_bound=vmax_bf_channel), cv2.COLOR_GRAY2BGR)
    result = cv2.add((red), (gray))
    return result
