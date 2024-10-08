# tests/unit/test_gpt2_service.py

import unittest
from src.service.gpt2_service import GPT2Service

class TestGPT2Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the service with GPT-2 for testing purposes
        cls.service = GPT2Service(model_name='gpt2')

    def test_gpt2_works(self):
        """
        Test that GPT-2 can receive requests and return responses.
        """
        user_id = 'user_test_1'
        input_json = {'input': 'Once upon a time'}
        response = self.service.process_request(user_id, input_json)

        # Assertions
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        self.assertNotEqual(response['response'].strip(), '')
        print('GPT-2 Works Test Response:', response['response'])

    def test_gpt2_json_handling(self):
        """
        Test that GPT-2 can receive and return JSON data.
        """
        user_id = 'user_test_2'
        input_json = {'input': 'In a galaxy far, far away'}
        response = self.service.process_request(user_id, input_json)

        # Assertions
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        self.assertNotEqual(response['response'].strip(), '')
        print('JSON Handling Test Response:', response['response'])

    def test_gpt2_user_sessions(self):
        """
        Test that GPT-2 distinguishes users and gives different answers
        based on previously given information.
        """
        user_id1 = 'user_alice'
        user_id2 = 'user_bob'

        # User Alice's interaction
        input_json_alice_1 = {'input': 'The secret to happiness is'}
        response_alice_1 = self.service.process_request(user_id1, input_json_alice_1)

        input_json_alice_2 = {'input': ' and also'}
        response_alice_2 = self.service.process_request(user_id1, input_json_alice_2)

        # User Bob's interaction
        input_json_bob_1 = {'input': 'The meaning of life is'}
        response_bob_1 = self.service.process_request(user_id2, input_json_bob_1)

        input_json_bob_2 = {'input': ' but sometimes'}
        response_bob_2 = self.service.process_request(user_id2, input_json_bob_2)

        # Assertions
        self.assertNotEqual(response_alice_2['response'], response_bob_2['response'])
        print('User Alice Response:', response_alice_2['response'])
        print('User Bob Response:', response_bob_2['response'])

        # Check that the sessions are stored separately
        self.assertNotEqual(
            self.service.user_sessions[user_id1],
            self.service.user_sessions[user_id2]
        )

if __name__ == '__main__':
    unittest.main()
