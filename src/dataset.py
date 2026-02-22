import torch as pt
from torch.utils.data import Dataset

class Patcher:
    def __init__(self, image, patch_size, patch_stride):
        self.image = image
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.grid = image.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride).permute(1, 2, 0, 3, 4)
        self.shape = self.grid.shape[:2]

    def __call__(self, coords):
        patches = self.grid[*coords.T]

        return patches


dumb_rng = lambda i, c, seed, x_min, x_max: (i ** c + seed) % (x_max - x_min) + x_min


def get_raw_path(i, num_points=128, c=51, seed=6, x_min=-5, x_max=5):
    # c and seed are genuinely divine numbers, beware when adjusting
    i = pt.arange(num_points * 2).view(-1, 2) + num_points * 2 * i

    dx = dumb_rng(i, c, seed, x_min, x_max)
    return pt.cat((pt.tensor([[0, 0]]), dx.cumsum(dim=0)))


def get_path(i, img_size, min_size, num_points=128, c=51, seed=6, x_min=-5, x_max=5):
    img_size = pt.tensor(img_size)
    raw_path = get_raw_path(i, num_points, c, seed, x_min, x_max)

    raw_min = raw_path.min(dim=0).values
    raw_max = raw_path.max(dim=0).values
    raw_range = raw_max - raw_min

    min_max = ((img_size - min_size) / 2).int()
    max_min = ((img_size + min_size) / 2).int()

    min_i, max_i = (pt.arange(4) + 4 * i).view(2, 2)
    path_min = dumb_rng(min_i, c, seed, 0, min_max)
    path_max = dumb_rng(max_i, c, seed, max_min, img_size)
    path_range = path_max - path_min
    print(img_size)
    print(path_max)

    path = (raw_path - raw_min) / raw_range * path_range + path_min
    return path.int()


class PathDataset(Dataset):
    def __init__(self, image, num_samples, num_points, patch_size=64, patch_stride=1, move_max=5):
        self.patcher = Patcher(image, patch_size, patch_stride)
        self.num_samples = num_samples
        self.num_points = num_points
        self.move_max = move_max

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        coords = get_path(idx, self.patcher.shape, self.num_points, x_min=-self.move_max, x_max=self.move_max)
        moves = coords.diff(dim=0)

        patches = self.patcher(coords)
        in_patches = patches[:-1]
        out_patches = patches[1:]

        return moves, in_patches, out_patches


class PatchDataset(Dataset):
    def __init__(self, image, patch_size=64, patch_stride=1):
        self.patcher = Patcher(image, patch_size, patch_stride)
        self.n_h, self.n_w = self.patcher.shape

    def __len__(self):
        return self.n_h * self.n_w

    def __getitem__(self, idx):
        y = idx // self.n_w
        x = idx % self.n_w
        coords = pt.tensor([y, x], dtype=pt.float32)
        patch = self.patcher(coords[None, :].int())[0]
        return patch, coords


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    image_path = 'frieren.png'
    print(f"Using image path: {image_path}")


    IMG = transforms.ToTensor()(transforms.Resize(512)(Image.open(image_path)))

    print(f"Original Image Size: {IMG.size} (Width, Height)")
    print(f"PyTorch Tensor Shape (C, H, W): {IMG.shape}")
    print(f"PyTorch Tensor Data Type: {IMG.dtype}")

    plt.imshow(IMG.permute(1, 2, 0))
    plt.title('Original Image from URL')
    plt.axis('off')
    plt.show()

    patcher = Patcher(IMG, 64, 16)
    ps = patcher(pt.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) + 8)

    for p in ps:
        plt.imshow(p.permute(1, 2, 0))
        plt.show()
        
    x = get_path(0, IMG.shape[1:], 128)

    plt.imshow(IMG.permute(1, 2, 0))
    plt.plot(x[:, 1], x[:, 0])
    plt.plot(x[0, 1], x[0, 0], 'gx')
    plt.show()
    
    a = PatchDataset(IMG)
    print("dataset length:", len(a))
