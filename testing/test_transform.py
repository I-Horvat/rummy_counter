import torch
from torchvision.transforms import v2

from card_dataset.CardDatsSet import CardDataset
from testing.plotting import plot_samples

num_of_pixels = 1024
# transform = v2.Compose([
#     v2.Resize(size=(num_of_pixels, num_of_pixels), antialias=True),
#     v2.RandomResizedCrop(size=(num_of_pixels, num_of_pixels), antialias=True),
#     v2.RandomPhotometricDistort(p=1),
#     v2.RandomHorizontalFlip(p=1),
# ])
# transform=None

# transform = v2.Compose([
#     v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
#     v2.RandomResizedCrop((num_of_pixels, num_of_pixels), antialias=True),
#     v2.RandomPhotometricDistort(p=1),
#     v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     v2.RandomRotation(10),
# ])
transform=None
# transform = v2.Compose([
#     v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
#     v2.RandomPhotometricDistort(p=1),
#     v2.RandomHorizontalFlip(p=1),
#     # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
transform = v2.Compose([
    v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

# transform = v2.Compose([
#     v2.ToImage(),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.RandomAdjustSharpness(sharpness_factor=2),
#     v2.ColorJitter(),
#     v2.Resize((num_of_pixels, num_of_pixels), antialias=True),
#     v2.ToDtype(torch.float32, scale=True),
# ])
# transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# card_dataset = CardDataset(root_dir='../../../data_augmentation/cleaned_and_total', transform=transform, num_of_pixels=num_of_pixels)
# card_dataset = CardDataset(root_dir='../../../data_augmentation/augmented', transform=transform, num_of_pixels=num_of_pixels)
card_dataset = CardDataset(root_dir='../dataset_generator/generated_dataset', transform=transform, num_of_pixels=num_of_pixels)


plot_samples(card_dataset,10)
#plot_sample_by_name(card_dataset, '0a246d3b-afb8-4c23-9dc3-2038ecb4a2b2')
# plot_sample_by_name(card_dataset, 'image_602')
# plot_sample_by_name(card_dataset, '24018d4f-87c4-4e00-b205-f2d3a5377ade')