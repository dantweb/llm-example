# tests/unit/test_langmodel_service.py

import unittest
from src.service.langmodel_service import LangModelService

class TestLangModelService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the service with DialoGPT-small for testing purposes
        cls.service = LangModelService(model_name='microsoft/DialoGPT-small')

    def test_llm_works(self):
        """
        Test that the LLM can receive requests and return responses.
        """
        user_id = 'user_test_1'
        input_json = {'input': 'Hello, how are you?'}
        response = self.service.process_request(user_id, input_json)

        # Assertions
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        print('LLM Works Test Response:', response['response'])

    def test_llm_json_handling(self):
        """
        Test that the LLM can receive and return JSON data.
        """
        user_id = 'user_test_2'
        input_json = {'input': 'What is the weather like today?'}
        response = self.service.process_request(user_id, input_json)

        # Assertions
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        print('JSON Handling Test Response:', response['response'])

    def test_llm_user_sessions(self):
        """
        Test that the LLM distinguishes users and gives different answers
        based on previously given information.
        """
        user_id1 = 'user_alice'
        user_id2 = 'user_bob'

        # User Alice's interaction
        input_json_alice_1 = {'input': 'My name is Alice.'}
        self.service.process_request(user_id1, input_json_alice_1)

        input_json_alice_2 = {'input': 'What is my name?'}
        response_alice = self.service.process_request(user_id1, input_json_alice_2)

        # User Bob's interaction
        input_json_bob_1 = {'input': 'My name is Bob.'}
        self.service.process_request(user_id2, input_json_bob_1)

        input_json_bob_2 = {'input': 'What is my name?'}
        response_bob = self.service.process_request(user_id2, input_json_bob_2)

        # Assertions
        self.assertNotEqual(response_alice['response'], response_bob['response'])
        print('User Alice Response:', response_alice['response'])
        print('User Bob Response:', response_bob['response'])

        # Check that the sessions are stored separately
        self.assertNotEqual(
            self.service.user_sessions[user_id1].tolist(),
            self.service.user_sessions[user_id2].tolist()
        )

if __name__ == '__main__':
    unittest.main()
