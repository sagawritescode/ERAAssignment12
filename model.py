class LitCustomResNet(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=16, learning_rate=2e-4, criterion=nn.CrossEntropyLoss(reduction="sum"), normalisation_method="bn", groups=0, means=[0.4914, 0.4822, 0.4465], stds=[0.2470, 0.2435, 0.2616]):
        super().__init__()

        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 64, groups),
            nn.ReLU(),
        ) # output_size = 32

        # Layer1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),
        ) # output_size = 16

        self.res_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),
        )

        # Layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 256, groups),
            nn.ReLU(),
        ) # output_size = 8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 512, groups),
            nn.ReLU(),
        ) # output_size = 4

        # ResBlock2 todo: make sure to add as different class as specified in description
        self.res_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 512, groups),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 512, groups),
            nn.ReLU(),
        ) # output_size = 4

        self.maxPool2 = nn.MaxPool2d(4, 4) # output_size = 1
        self.output_linear = nn.Linear(512, 10, bias=False)
        self.accuracy = Accuracy("multiclass", num_classes=10)
        self.save_hyperparameters()
        self.misclassified_images = []
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
        self.means = means
        self.stds = stds
        self.train_transforms = CustomResnetTransforms.train_transforms(means, stds)
        self.test_transforms = CustomResnetTransforms.test_transforms(means, stds)

        # Create the reverse transformation pipeline
        self.reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-mean / std for mean, std in zip(means, stds)],
                                std=[1 / std for std in stds]),
            transforms.ToPILImage()
        ])


    def forward(self, x):
        x = self.prep_layer(x)
        x = self.convblock1(x)
        x = x + self.res_block1(x)
        x = self.layer2(x)
        x = self.convblock2(x)
        x = x + self.res_block2(x)
        x = self.maxPool2(x)
        x = x.view(x.size(0), -1)
        x = self.output_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print("printing shape: ", x.shape)
        # print("printing shape: ", x.shape, y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("training_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        for i in range(len(y)):
            if preds[i] != y[i]:
                self.misclassified_images.append((x[i], y[i], preds[i]))
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        # Save the misclassified images at the end of the validation epoch
        for i, misclassified_image_tuple in enumerate(self.misclassified_images):
            image_tensor, correct, wrong = misclassified_image_tuple
            image_pil = self.reverse_transform(image_tensor.cpu())
            image_pil.save(f'{self.data_dir}/misclassified_images/misclassified_image_{self.classes[correct]}_{self.classes[wrong]}.png')

    def prepare_data(self):
      # download
      Cifar10SearchDataset(self.data_dir, train=True, download=True)
      Cifar10SearchDataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_train = Cifar10SearchDataset(self.data_dir, train=True, transform=self.train_transforms)
            

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = Cifar10SearchDataset(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count())