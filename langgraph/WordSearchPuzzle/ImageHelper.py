from PIL import Image
import base64
from io import BytesIO
import logging
import sys

class ImageHelper: 
    #def _init_(self,filePath:str):
            #image_path = 'path/to/your/image.jpg'
            
    def ReadImageAsBase64String(filePath:str) -> str:
        image = Image.open(filePath,"r",)
        buffered = BytesIO()
        image.save(buffered) #, format="JPEG" Or PNG, depending on your image type
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
    
    def image_to_base64_string(image_path:str) -> str:
        """
        Reads an image file and returns its Base64 encoded string representation.
        Args:            
            image_path (str): The path to the image file.
        Returns:
            str: The Base64 encoded string of the image, or None if an error occurs.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_binary_data = image_file.read()
                base64_encoded_bytes = base64.b64encode(image_binary_data)
                base64_string = base64_encoded_bytes.decode("utf-8")
                return base64_string
        except FileNotFoundError:
            print(f"Error: Image file not found at '{image_path}'")
            return None
        except Exception as e:
            logger.error(f"Error in extracting char array from image: {str(e)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)