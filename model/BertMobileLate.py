import torch.nn as nn
from transformers import BertModel
import torchvision

class BertMobileLate(nn.Module):
    def __init__(self):
        super(BertMobileLate, self).__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')  
        self.img_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.t_linear = nn.Linear(768, 128)  
        self.i_linear = nn.Linear(1000, 128)
        self.fc_txt = nn.Linear(128, 3)  # 用于文本的最终线性层
        self.fc_img = nn.Linear(128, 3)  # 用于图像的最终线性层
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        # 获取文本输出
        txt_out = self.t_linear(self.txt_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])
        txt_out = self.relu(txt_out)
        
        # 获取图像输出
        img_out = self.i_linear(self.img_model(image))
        img_out = self.relu(img_out)
        
        # 经过各自的最终线性层
        txt_out_final = self.fc_txt(txt_out)
        img_out_final = self.fc_img(img_out)
        
        # 对文本和图像的最终输出进行平坄1�7
        avg_out = (txt_out_final + img_out_final) / 2.0
        
        return avg_out
