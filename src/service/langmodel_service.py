# src/service/langmodel_service.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LangModelService:
    def __init__(self, model_name='microsoft/DialoGPT-small'):
        """
        Initialize the language model service with the specified model.
        Using 'microsoft/DialoGPT-small' for conversational capabilities.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.user_sessions = {}  # Dictionary to hold user sessions

    def process_request(self, user_id, input_json):
        """
        Process a user's request and generate a response.

        Args:
            user_id (str): Unique identifier for the user.
            input_json (dict): JSON object containing the user's input.

        Returns:
            dict: JSON response containing the assistant's reply.
        """
        # Extract input text from JSON
        input_text = input_json.get('input', '')
        if not input_text:
            return {'error': 'No input text provided'}

        # Retrieve user's chat history
        chat_history_ids = self.user_sessions.get(user_id, None)

        # Encode the user's input and append EOS token
        new_user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

        # Append the new user input to the chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate the assistant's response
        chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=bot_input_ids.shape[-1] + 50,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        # Decode the generated response
        assistant_reply = self.tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        # Update the user's chat history
        self.user_sessions[user_id] = chat_history_ids

        # Return the assistant's reply as a JSON object
        response_json = {'response': assistant_reply}
        return response_json
