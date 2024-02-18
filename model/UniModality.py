import torch.nn as nn
from transformers import BertModel
import torchvision

class UniModality(nn.Module):
    def __init__(self, in_features, out_features, pretrain):
        super(UniModality, self).__init__()
        if pretrain == "bert":
            self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.img_model = torchvision.models.resnet152(pretrained=True)
        self.linear = nn.Linear(in_features, out_features)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        if hasattr(self, 'txt_model'):
            x = self.txt_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        else:
            x = self.img_model(image)

        x = self.linear(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class TxtModel(UniModality):
    def __init__(self):
        super(TxtModel, self).__init__(768, 256, "bert")


class ImgModel(UniModality):
    def __init__(self):
        super(ImgModel, self).__init__(1000, 256, "resnet")

