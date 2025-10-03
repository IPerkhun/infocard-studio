from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_PATH = "models/qwen_detect_product"

class QwenVLModel:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()

    def get_model(self):
        return self.model, self.processor
