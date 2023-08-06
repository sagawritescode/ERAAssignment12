import albumentations as A 
from albumentations.pytorch import ToTensorV2 

class CustomResnetTransforms:
    def train_transforms(means, stds):
        return A.Compose(
            [
                A.Normalize(mean=means, std=stds),
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )

    def test_transforms(means, stds):
        return A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2()
            ]
        )
