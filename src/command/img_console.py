import sys
import os
import uuid
import argparse
from pathlib import Path

# Adjust the import path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_unique_filename(directory, filename):
    """
    Get a unique filename by appending an index if the file already exists.

    Args:
        directory (str): The directory where the file will be saved.
        filename (str): The desired filename.

    Returns:
        str: A unique filename with an index if necessary.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def main():
    """
    Main function to run the command-line image generator interface.
    """
    # Ensure the _generated directory exists
    output_dir = Path('./_data/_generated')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate an image using a locally hosted AI model.')
    parser.add_argument('--prompt', type=str, help='Description of the image to generate', required=True)
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps (default: 50)')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for generation (default: 7.5)')
    parser.add_argument('--filename', type=str, default='generated_image.png', help='Filename for the generated image (default: generated_image.png)')
    parser.add_argument('--width', type=int, default=512, help='Width of the generated image (default: 512)')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated image (default: 512)')
    parser.add_argument('--imgformat', type=str, choices=['jpg', 'png'], default='png', help='Image format (default: png)')

    args = parser.parse_args()

    # Import the ImageGenService
    from service.imagegen_service import ImageGenService

    # Generate a unique user ID for the session
    user_id = str(uuid.uuid4())

    # Initialize the image generation service
    service = ImageGenService()

    # Set the image size and format using setters
    service.picture_size = {'width': args.width, 'height': args.height}
    service.imgformat = args.imgformat

    # Prepare input JSON with user prompt and optional parameters
    input_json = {
        'prompt': args.prompt,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale
    }

    # Get a unique filename if the file already exists
    unique_filename = get_unique_filename(output_dir, args.filename)
    file_path = output_dir / unique_filename

    # Process the image generation request
    result = service.process_request(user_id, input_json)

    # Move the generated image to the correct path
    if 'image_file_path' in result:
        os.rename(result['image_file_path'], file_path)
        print(f"Image generated and saved to {file_path}")
    else:
        print(f"Error: {result.get('error', 'Unknown error occurred')}")


if __name__ == '__main__':
    main()
