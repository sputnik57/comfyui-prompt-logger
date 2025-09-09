import unittest
from unittest.mock import MagicMock
from PromptLoggerUnified_2 import PromptLoggerUnified

class TestPromptLoggerUnified(unittest.TestCase):

    def setUp(self):
        self.logger = PromptLoggerUnified()

    def test_extract_model_metadata(self):
        model_mock = MagicMock()
        model_mock.model.config = {'model_type': 'SDXL'}
        model_mock.model.state_dict.return_value = {'key': torch.tensor([1.0])}

        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertEqual(metadata, {'model_type': 'SDXL', 'model_hash': '627fcf8b'})

        model_mock.model.config = {'in_channels': 9}
        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertEqual(metadata, {'model_type': 'SD_Inpainting', 'model_hash': '627fcf8b'})

        model_mock.model.config = {'model_channels': 320, 'in_channels': 4}
        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertEqual(metadata, {'model_type': 'SDXL', 'model_hash': '627fcf8b'})

        model_mock.model.config = {}
        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertEqual(metadata, {'model_type': 'SD1.5', 'model_hash': '627fcf8b'})

        model_mock.model.config = {'model_type': 'Unknown'}
        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertEqual(metadata, {'model_type': 'Unknown', 'model_hash': '627fcf8b'})

        model_mock.model = None
        metadata = self.logger.extract_model_metadata(model_mock)
        self.assertIsNone(metadata)

    def test_parse_lora_info(self):
        lora_info_str = "lora1:0.5:0.7\nlora2:0.8"
        lora_list = self.logger.parse_lora_info(lora_info_str)
        self.assertEqual(lora_list, [
            {'name': 'lora1', 'strength_model': 0.5, 'strength_clip': 0.7},
            {'name': 'lora2', 'strength_model': 0.8, 'strength_clip': 0.8}
        ])

        lora_info_str = "lora3"
        lora_list = self.logger.parse_lora_info(lora_info_str)
        self.assertEqual(lora_list, [
            {'name': 'lora3', 'strength_model': 1.0, 'strength_clip': 1.0}
        ])

        lora_info_str = ""
        lora_list = self.logger.parse_lora_info(lora_info_str)
        self.assertEqual(lora_list, [])

    def test_log_and_generate(self):
        # Mocking the necessary components
        model_mock = MagicMock()
        model_mock.model.config = {'model_type': 'SDXL'}
        model_mock.model.state_dict.return_value = {'key': torch.tensor([1.0])}

        # Call the method
        result = self.logger.log_and_generate(
            prompt="Test prompt",
            folder="output/test",
            base_name="test_log",
            sampler="euler",
            scheduler="normal",
            denoise=1.0,
            steps=20,
            cfg=7.5,
            seed=2025,
            use_timestamp=True,
            timestamp_format="%d%b%Y_%H%M",
            control_after_generate=False,
            model=model_mock,
            checkpoint_name="checkpoint1",
            lora_info="lora1:0.5:0.7\nlora2:0.8",
            vae_name="vae1"
        )

        # Verify the result
        self.assertEqual(result, (
            "Test prompt",
            "output/test/test_log_09Sep2025_1510.png",
            "euler",
            "normal",
            20,
            7.5,
            2025,
            1.0,
            False
        ))

        # Verify the JSON file content
        with open("output/test/test_log_09Sep2025_1510.json", "r", encoding="utf-8") as f:
            entry = json.load(f)
            self.assertEqual(entry, {
                "filename": "test_log_09Sep2025_1510.png",
                "timestamp": "2025-09-09T15:10:00-07:00",
                "prompt": "Test prompt",
                "folder": "output/test",
                "base_name": "test_log",
                "sampler": "euler",
                "scheduler": "normal",
                "steps": 20,
                "cfg": 7.5,
                "seed": 2025,
                "control_after_generate": False,
                "denoise": 1.0,
                "use_timestamp": True,
                "timestamp_format": "%d%b%Y_%H%M",
                "ksampler": {
                    "sampler": "euler",
                    "scheduler": "normal",
                    "steps": 20,
                    "cfg": 7.5,
                    "seed": 2025
                },
                "models": {
                    "checkpoint": "checkpoint1",
                    "model_type": "SDXL",
                    "model_hash": "627fcf8b",
                    "loras": [
                        {'name': 'lora1', 'strength_model': 0.5, 'strength_clip': 0.7},
                        {'name': 'lora2', 'strength_model': 0.8, 'strength_clip': 0.8}
                    ],
                    "vae": "vae1"
                }
            })

if __name__ == '__main__':
    unittest.main()
