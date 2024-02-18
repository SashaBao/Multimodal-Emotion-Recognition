import torch
import torch.nn as nn
from transformers import BertModel
import torchvision

class BertResEarly(nn.Module):
    def __init__(self):
        super(BertResEarly, self).__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.img_model = torchvision.models.resnet152(pretrained=True)
        self.t_linear = nn.Linear(768, 128)
        self.i_linear = nn.Linear(1000, 128)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.i_linear(self.img_model(image))
        txt_out = self.t_linear(self.txt_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])
        last_out = self.fc(self.relu(torch.cat((txt_out, img_out), dim=-1)))
        return last_out