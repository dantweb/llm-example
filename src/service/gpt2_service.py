# src/service/gpt2_service.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

class GPT2Service:
    def __init__(self, model_name='gpt2'):
        """
        Initialize the GPT-2 service with the specified model.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.user_sessions = {}  # Dictionary to hold user sessions

    def process_request(self, user_id, input_json):
        """
        Process a user's request and generate a response.

        Args:
            user_id (str): Unique identifier for the user.
            input_json (dict): JSON object containing the user's input.

        Returns:
            dict: JSON response containing the generated text.
        """
        # Extract input text from JSON
        input_text = input_json.get('input', '')
        if not input_text:
            return {'error': 'No input text provided'}

        # Retrieve user's past inputs
        user_history = self.user_sessions.get(user_id, '')

        # Concatenate past inputs with current input
        full_input = user_history + input_text

        # Encode the input text
        input_ids = self.tokenizer.encode(full_input, return_tensors='pt')

        # Generate the model's output
        output_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + 50,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        # Decode the output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Update the user's session history
        self.user_sessions[user_id] = generated_text

        # Return the generated text as a JSON object
        response_json = {'response': generated_text[len(user_history):]}
        return response_json
