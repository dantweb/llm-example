# src/command/console.py

import sys
import os
import uuid
import argparse

# Adjust the import path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    Main function to run the command-line chat interface.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Chat with a language model.')
    parser.add_argument(
        '--service',
        choices=['gpt2_service', 'langmodel_service'],
        default='langmodel_service',
        help='Choose the language model service to use.'
    )
    args = parser.parse_args()

    # Import the chosen service
    if args.service == 'gpt2_service':
        from service.gpt2_service import GPT2Service as SelectedService
        model_name = 'gpt2'
    elif args.service == 'langmodel_service':
        from service.langmodel_service import LangModelService as SelectedService
        model_name = 'microsoft/DialoGPT-small'

    # Generate a unique user ID for the session
    user_id = str(uuid.uuid4())

    # Initialize the selected language model service
    service = SelectedService(model_name=model_name)

    print(f"Welcome to the Chatbot using {args.service}! Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        # Prepare input JSON
        input_json = {'input': user_input}

        # Get response from the language model service
        response_json = service.process_request(user_id, input_json)

        # Handle possible errors
        if 'error' in response_json:
            print(f"Error: {response_json['error']}")
        else:
            assistant_reply = response_json.get('response', '')
            print(f"Chatbot: {assistant_reply}")

if __name__ == '__main__':
    main()
