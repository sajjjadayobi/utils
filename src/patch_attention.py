import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = 'google/vit-base-patch16-224'
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

def concate_two_image_patches(images, sep_emb):
    inputs = feature_extractor(images=images, return_tensors='pt')
    images = inputs['pixel_values'].unsqueeze(dim=1)
    img1 = model.base_model.embeddings(images[0])
    img2 = model.base_model.embeddings(images[1])
    return torch.concatenate((img1, sep_emb, img2), dim=1)

images = torch.rand((2, 3, 224, 224))
sep_emb = torch.ones((1, 1, 768))
cated_images = concate_two_image_patches(images, sep_emb)

print(cated_images.shape)
print(model.base_model.encoder(cated_images).last_hidden_state.shape)
