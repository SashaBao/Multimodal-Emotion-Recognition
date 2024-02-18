import torch.nn as nn
from transformers import BertModel
import torchvision


class BertMobileTensor(nn.Module):
    def __init__(self):
        super(BertMobileTensor, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.text_linear = nn.Linear(768, 128)
        self.image_linear = nn.Linear(1000, 128)
        self.image_gate = nn.Linear(128, 1)
        self.text_gate = nn.Linear(128, 1)
        self.final_linear = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        # 澶勭悊鍥惧儚
        image_out = self.image_model(image)
        image_out = self.image_linear(image_out)
        image_out = self.relu(image_out)
        image_weight = self.image_gate(image_out)

        # 澶勭悊鏂囨湰
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_out = self.text_linear(text_out)
        text_out = self.relu(text_out)
        text_weight = self.text_gate(text_out)

        # 鏉冮噸铻嶅悎
        fused_out = image_weight * image_out + text_weight * text_out

        # 鏈€缁堣緭鍑�
        final_out = self.final_linear(fused_out)
        return final_out
