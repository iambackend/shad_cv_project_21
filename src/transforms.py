from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)), # Hint: this might not be the best way to resize images
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]) # Hint: this might not be the best normalization
     ]
)
