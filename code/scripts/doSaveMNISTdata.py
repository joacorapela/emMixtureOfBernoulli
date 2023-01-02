import sys
import torch
import pandas as pd
import torchvision


def test_done(saved_per_digit, num_per_digit):
    for n in saved_per_digit:
        if n < num_per_digit:
            return False
    return True


def get_data(enum, digits, num_per_digit, image_width=28, image_height=28, ):
    n_data_points = len(digits)*num_per_digit
    images = []
    targets = []
    saved_per_digit = [0 for i in range(len(digits))]
    done = test_done(saved_per_digit=saved_per_digit,
                     num_per_digit=num_per_digit)
    i = 0
    while not done:
        batch_idx, (enum_images, enum_targets) = next(enum)
        try:
            index = digits.index(enum_targets[0])
        except ValueError:
            continue
        if saved_per_digit[index] < num_per_digit:
            saved_per_digit[index] += 1
            image_flatten_grayscale = enum_images[0].flatten()
            image_flatten_binary = torch.where(
                image_flatten_grayscale>0.5, 1, 0).tolist()
            images.append(image_flatten_binary)
            targets.append(enum_targets[0].item())
        done = test_done(saved_per_digit=saved_per_digit,
                         num_per_digit=num_per_digit)
    images_t = torch.tensor(images)
    targets_t = torch.tensor(targets)
    return images_t, targets_t


def save_tensor_to_csv(tensor, filename):
    tensor_np = tensor.numpy()
    tensor_df = pd.DataFrame(tensor_np)
    tensor_df.to_csv(filename, header=False, index=False, sep=' ')


def main(argv):
    digits = [2, 3, 4]
    num_train_per_digit = 200
    num_test_per_digit = 100
    random_seed = 1

    torch.manual_seed(random_seed)

    train_mnist_data = torchvision.datasets.MNIST('../../data/mnist',
                                                  train=True, download=True,
                                                  transform = torchvision.transforms.Compose([
                                                      # you can add other
                                                      # transformations in this
                                                      # list
                                                      torchvision.transforms.ToTensor(),
                                                  ]))
    train_loader = torch.utils.data.DataLoader(train_mnist_data)
    train_enum = enumerate(train_loader)
    train_images, train_targets = get_data(enum=train_enum, digits=digits,
                                          num_per_digit=num_train_per_digit )

    test_mnist_data = torchvision.datasets.MNIST('../../data/mnist',
                                                 train=False, download=True,
                                                 transform = torchvision.transforms.Compose([
                                                     # you can add other
                                                     # transformations in this
                                                     # list
                                                     torchvision.transforms.ToTensor(),
                                                  ]))
    test_loader = torch.utils.data.DataLoader(test_mnist_data)
    test_enum = enumerate(test_loader)
    test_images, test_targets = get_data(enum=test_enum, digits=digits,
                                        num_per_digit=num_test_per_digit)

    digits_str = "_".join([str(i) for i in digits])
    save_tensor_to_csv(tensor=train_images,
                      filename=f'../../data/train_mnist_images_digits{digits_str}_numPerDigit{num_train_per_digit}.csv')
    save_tensor_to_csv(tensor=train_targets,
                      filename=f'../../data/train_mnist_targets_digits{digits_str}_numPerDigit{num_train_per_digit}.csv')
    save_tensor_to_csv(tensor=test_images,
                      filename=f'../../data/test_mnist_images_digits{digits_str}_numPerDigit{num_test_per_digit}.csv')
    save_tensor_to_csv(tensor=test_targets,
                      filename=f'../../data/test_mnist_targets_digits{digits_str}_numPerDigit{num_test_per_digit}.csv')
    breakpoint()


if __name__ == '__main__':
    main(sys.argv)
