import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, XLNetTokenizer
import argparse

from model.BertResEarly import BertResEarly
from model.BertResLate import BertResLate
from model.BertResTensor import BertResTensor
from model.BertMobileEarly import BertMobileEarly
from model.BertMobileLate import BertMobileLate
from model.BertMobileTensor import BertMobileTensor
from model.XLNetResEarly import XLNetResEarly
from model.XLNetResLate import XLNetResLate
from model.XLNetResTensor import XLNetResTensor
from model.XLNetMobileEarly import XLNetMobileEarly
from model.XLNetMobileLate import XLNetMobileLate
from model.XLNetMobileTensor import XLNetMobileTensor
from model.UniModality import TxtModel, ImgModel
from utils.generate_dataset import CustomDataset, generate_input_ids, generate_mask
from utils.train import train

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_encoder', default='bert')
    parser.add_argument('--image_encoder', default='resnet')
    parser.add_argument('--fusion_method', default='early')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch', default=20, type=int)

    model_classes = {
        ('bert', 'resnet', 'early'): BertResEarly,
        ('bert', 'resnet', 'late'): BertResLate,
        ('bert', 'resnet', 'tensor'): BertResTensor,
        ('bert', 'mobilenet', 'early'): BertMobileEarly,
        ('bert', 'mobilenet', 'late'): BertMobileLate,
        ('bert', 'mobilenet', 'tensor'): BertMobileTensor,
        ('xlnet', 'resnet', 'early'): XLNetResEarly,
        ('xlnet', 'resnet', 'late'): XLNetResLate,
        ('xlnet', 'resnet', 'tensor'): XLNetResTensor,
        ('xlnet', 'mobilenet', 'early'): XLNetMobileEarly,
        ('xlnet', 'mobilenet', 'late'): XLNetMobileLate,
        ('xlnet', 'mobilenet', 'tensor'): XLNetMobileTensor,
    }

    args = parser.parse_args()

    model_key = (args.text_encoder, args.image_encoder, args.fusion_method)
    model_class = model_classes.get(model_key)

    if args.image_only:
        model = ImgModel().to(device)
    elif args.text_only:
        model = TxtModel().to(device)
    else:
        if model_class:
            model = model_class().to(device)
        else:
            print("Invalid combination of text_encoder, image_encoder, and fusion_method.")

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epoch = args.epoch
    if args.text_encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.text_encoder == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet/xlnet-base-cased')

    df = pd.read_csv("./data/text_tag_train.csv", keep_default_na=False)

    X = df[['guid', 'text']]
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    images_train, images_valid = [], []
    for guid in X_train['guid']:
        pil_image = Image.open(f"./data/data/{guid}.jpg").resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(pil_image, dtype='float32')
        images_train.append(img.transpose(2, 0, 1))

    for guid in X_test['guid']:
        pil_image = Image.open(f"./data/data/{guid}.jpg").resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(pil_image, dtype='float32')
        images_valid.append(img.transpose(2, 0, 1))

    input_ids_train, input_ids_valid = [], []
    input_ids_train = generate_input_ids(X_train['text'], tokenizer)
    input_ids_valid = generate_input_ids(X_test['text'], tokenizer)

    attention_mask_train, attention_mask_valid = [], []
    attention_mask_train = generate_mask(X_train['text'], tokenizer)
    attention_mask_valid = generate_mask(X_test['text'], tokenizer)

    batch_size = 16
    shuffle = True

    custom_dataset = CustomDataset(images_train, y_train, input_ids_train, attention_mask_train)
    train_data = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    custom_dataset = CustomDataset(images_valid, y_test, input_ids_valid, attention_mask_valid)
    valid_data = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


    train(model, epoch, optimizer, train_data, valid_data, device)
    
    # 预测
    emotion_labels = ["neutral", "negative", "positive"]
    test_data = pd.read_csv("./data/test_without_label.txt")
    guid_list = test_data['guid'].tolist()
    
    model_path = f'./saved_model/best_{model.__class__.__name__}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predicted_tags = []

    for guid in guid_list:
        image_path = f'./data/data/{guid}.jpg'
        img = Image.open(image_path).resize((224, 224), Image.Resampling.LANCZOS)
        image_tensor = np.asarray(img, dtype='float32').transpose(2, 0, 1)

        text_path = f'./data/data/{guid}.txt'
        with open(text_path, encoding='gb18030') as file:
            text = file.read()

        result = tokenizer([text], truncation=True, padding='max_length', max_length=32, return_tensors='pt')
        input_ids, attention_mask = result['input_ids'], result['attention_mask']

        predictions = model(input_ids.to(device), attention_mask.to(device), torch.Tensor(image_tensor).unsqueeze(0).to(device))

        predicted_tag = emotion_labels[predictions.argmax(dim=-1).item()]

        predicted_tags.append(predicted_tag)

    result_df = pd.DataFrame({'guid': guid_list, 'tag': predicted_tags})
    result_df.to_csv(f'./prediction/{model.__class__.__name__}.txt', sep=',', index=False)