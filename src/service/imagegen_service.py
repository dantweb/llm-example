import os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline


class ImageGenService:
    # Static properties (class-level attributes)
    DEFAULT_MODEL_NAME = 'CompVis/stable-diffusion-v1-4'
    DEFAULT_OUTPUT_DIR = './generated_images'
    DEFAULT_PICTURE_SIZE = {'width': 512, 'height': 512}
    DEFAULT_IMG_FORMAT = 'png'
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.5

    def __init__(self, model_name=None, output_dir=None, device=None, picture_size=None, imgformat=None):
        """
        Initialize the image generation service.

        Args:
            model_name (str): Name of the model to load.
            output_dir (str): Directory where generated images will be saved.
            device (str): The device to run the model on ('cuda' or 'cpu').
            picture_size (dict): Dictionary specifying 'width' and 'height' of the image.
            imgformat (str): Image format ('jpg' or 'png').
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.device = device or self.DEFAULT_DEVICE
        self.picture_size = picture_size or self.DEFAULT_PICTURE_SIZE
        self.imgformat = imgformat or self.DEFAULT_IMG_FORMAT

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_name).to(self.device)
        self.user_requests = {}  # Dictionary to hold user request history

    @property
    def picture_size(self):
        return self._picture_size

    @picture_size.setter
    def picture_size(self, size_dict):
        if isinstance(size_dict, dict) and 'width' in size_dict and 'height' in size_dict:
            self._picture_size = size_dict
        else:
            raise ValueError("picture_size must be a dictionary with 'width' and 'height' keys.")

    @property
    def imgformat(self):
        return self._imgformat

    @imgformat.setter
    def imgformat(self, format_str):
        if format_str.lower() in ['jpg', 'png']:
            self._imgformat = format_str.lower()
        else:
            raise ValueError("imgformat must be either 'jpg' or 'png'.")

    def _generate_image(self, prompt_text, num_inference_steps, guidance_scale):
        """
        Generate an image using the Stable Diffusion pipeline.

        Args:
            prompt_text (str): The prompt text for image generation.
            num_inference_steps (int): Number of inference steps.
            guidance_scale (float): Guidance scale for image generation.

        Returns:
            Image: The generated image.
        """
        with torch.no_grad():
            image = \
            self.pipeline(prompt_text, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

            # Resize the image according to picture_size property
            image = image.resize((self.picture_size['width'], self.picture_size['height']))
            return image

    def _get_unique_filename(self, user_id):
        """
        Generate a unique filename by appending an index if the file already exists.

        Args:
            user_id (str): Unique identifier for the user.

        Returns:
            str: A unique file path for the generated image.
        """
        base_filename = f'{user_id}_generated_image.{self.imgformat}'
        file_path = os.path.join(self.output_dir, base_filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.output_dir, f'{user_id}_generated_image_{counter}.{self.imgformat}')
            counter += 1
        return file_path

    def _save_image(self, image, file_path):
        """
        Save the generated image to the specified file path.

        Args:
            image (Image): The generated image.
            file_path (str): Path where the image should be saved.
        """
        image.save(file_path, format=self.imgformat.upper())

    def process_request(self, user_id, input_json):
        """
        Process a user's image generation request and generate an image.

        Args:
            user_id (str): Unique identifier for the user.
            input_json (dict): JSON object containing the user's input.

        Returns:
            dict: JSON response containing the file path to the generated image.
        """
        prompt_text = input_json.get('prompt', '')
        if not prompt_text:
            return {'error': 'No prompt provided'}

        num_inference_steps = input_json.get('num_inference_steps', self.DEFAULT_NUM_INFERENCE_STEPS)
        guidance_scale = input_json.get('guidance_scale', self.DEFAULT_GUIDANCE_SCALE)

        try:
            image = self._generate_image(prompt_text, num_inference_steps, guidance_scale)
            image_file_path = self._get_unique_filename(user_id)
            self._save_image(image, image_file_path)
            self.user_requests[user_id] = {
                'prompt': prompt_text,
                'image_file_path': image_file_path
            }

            return {'image_file_path': image_file_path}

        except Exception as e:
            return {'error': f'Image generation failed: {str(e)}'}
