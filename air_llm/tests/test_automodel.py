import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..airllm.auto_model import AutoModel


class TestAutoModel(unittest.TestCase):
    def get_module_class_for_arch(self, architecture):
        config = SimpleNamespace(architectures=[architecture] if architecture else [])
        with patch("air_llm.airllm.auto_model.AutoConfig.from_pretrained", return_value=config):
            return AutoModel.get_module_class("test/model")

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            "QWenLMHeadModel": "AirLLMQWen",
            "InternLMForCausalLM": "AirLLMInternLM",
            "ChatGLMModel": "AirLLMChatGLM",
            "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
            "BaichuanForCausalLM": "AirLLMBaichuan",
            "BaiChuanForCausalLM": "AirLLMBaichuan",
        }

        for architecture, expected_class in mapping_dict.items():
            with self.subTest(architecture=architecture):
                module, cls = self.get_module_class_for_arch(architecture)
                self.assertEqual(module, "airllm")
                self.assertEqual(cls, expected_class, f"expecting {expected_class}")

    def test_standard_architectures_should_use_generic_model(self):
        for architecture in ["LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", None]:
            with self.subTest(architecture=architecture):
                module, cls = self.get_module_class_for_arch(architecture)
                self.assertEqual(module, "airllm")
                self.assertEqual(cls, "AirLLMBaseModel")

