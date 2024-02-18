from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_data, tags, inputs, masks):
        self.image_data = image_data
        if type(tags) != type([]):
            self.tags = tags.tolist()
        self.input_ids_data = inputs
        self.attention_mask_data = masks
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        tag = self.tags[index]
        input_ids = self.input_ids_data[index]
        attention_mask = self.attention_mask_data[index]

        return image, tag, input_ids, attention_mask

def generate_input_ids(text, tokenizer):
    text = text.tolist()
    result = tokenizer(text, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
    return result['input_ids']

def generate_mask(text, tokenizer):
    text = text.tolist()
    result = tokenizer(text, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
    return result['attention_mask']