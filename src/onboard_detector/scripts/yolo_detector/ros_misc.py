import sys
import numpy as np
from sensor_msgs.msg import Image
import rospy


def imgmsg_to_cv2(img_msg):
    if img_msg.encoding == "mono8":
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
    elif img_msg.encoding == "mono16":
        dtype = np.dtype("uint16")
    elif img_msg.encoding == "bgr8":
        dtype = np.dtype("uint8")
    else:
        raise TypeError("Unsupported encoding: %s" % img_msg.encoding)
    
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    
    if img_msg.encoding == "bgr8":
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # 3 channels for BGR
                        dtype=dtype, buffer=img_msg.data)
    else:
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width), # No channel dimension for mono images
                        dtype=dtype, buffer=img_msg.data)
    
    # If the byte order is different between the message and the system.
    if img_msg.is_bigendian != (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    
    return image_opencv

def cv2_to_imgmsg(cv_image, encoding="mono8"):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg
