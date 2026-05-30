import matplotlib.pyplot as plt
from pathlib import Path

from dataset import CamusPatientDataset

root = Path("C:\\Users\\Salem\\Documents\\Projects\\camus-segmentation-pytorch\\database_nifti")

patient_id = "patient0001"

dataset = CamusPatientDataset(
    patients=[patient_id],
    nifti_root=root,
    image_size=(256,256),
    augment=False
)

print(f"Number of samples: {len(dataset)}")

for i in range(len(dataset)):

    image, mask = dataset[i]

    print(f"\nSample {i}")
    print(f"Labels: {image.shape}")
    print(f"Mask labels: {mask.unique()}")

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")

    plt.show()