import torch
from torchvision import models
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import morphology
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import pickle


def main(
    image_path="data/image_samples/cat.jpg",
    kernel_size=3,
    nblocks=3,
    nlayers=3,
    epochs=500,
    lr=0.01,
    l1_coeff=1.0,
    indicate_center_of_mass=False,
):
    vgg_model = load_vgg_model()
    img = load_image(image_path)
    img_tensor = image_to_vgg_input_tensor(img)
    vgg_input_assessment(img_tensor, vgg_model)
    pert_model = PerturbationsGenerator(
        kernel_size, nblocks, nlayers,
    )
    pert_img_tensor = get_optimum_perturbation(
        epochs, pert_model, img_tensor,
        vgg_model=vgg_model,
        lr=lr, l1_coeff=l1_coeff,
    )
    diff, proc_img_np, pert_img_np = post_processing(
        img_tensor, pert_img_tensor,
    )
    plot_results(
        proc_img_np, pert_img_np, diff,
        indicate_center_of_mass=indicate_center_of_mass,
    )

    return proc_img_np, pert_img_np, diff


class PerturbationsGenerator(torch.nn.Module):
    def __init__(self, kernel_size=3, nblocks=3, nlayers=3):
        super(PerturbationsGenerator, self).__init__()
        # build conv layers, implement padding='same':
        if np.mod(kernel_size, 2) == 0: kernel_size += 1
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            3, 3, kernel_size = kernel_size,
            padding = padding,
        )
        self.relu = torch.nn.ReLU()
        self.nblocks = nblocks
        self.nlayers = nlayers

        if use_cuda(): self.cuda()

    def forward(self, x):
        # gather information for scaling
        xmin = torch.min(x)
        Dx = torch.max(x - xmin)

        # perturbate the image:
        for __ in range(self.nblocks):
            for __ in range(self.nlayers):
                x = self.conv(x)
            x = self.relu(x)

        # scale to original input range:
        x = x.add(- torch.min(x))  # x: zero to something
        x = x.div(torch.max(x))  # x: zero to 1
        x = x.mul(Dx)  # x: zero to Dx
        x = x.add(xmin)  # x: xmin to xmin + Dx

        if use_cuda(): x.cuda()

        return x


def get_optimum_perturbation(
        epochs, pert_model, img, vgg_model,
        lr=0.1, l1_coeff=0.01,
):
    optimizer = torch.optim.Adam(
        pert_model.parameters(), lr=lr
    )
    target = torch.nn.Softmax()(vgg_model(img))
    category = np.argmax(target.cpu().data.numpy())
    print "Category with highest probability", category
    print "Optimizing.. "
    losses = []

    for i in tqdm(range(epochs)):
        pert_img = pert_model(img)
        outputs = torch.nn.Softmax()(vgg_model(pert_img))
        img_diff = img - pert_img
        l1_term = l1_coeff * torch.mean(torch.abs(torch.pow(img_diff, 1)))
        loss = l1_term + outputs[0, category]
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot the loss:
    plt.figure("loss")
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")

    print "original score: {}".format(torch.nn.Softmax()(vgg_model(img))[0, category])
    print "perturbed score: {}".format(torch.nn.Softmax()(vgg_model(pert_img))[0, category])

    return pert_img


def load_image(image_path, graph=False):
    img = io.imread(image_path)

    if graph:
        plt.figure("original image")
        plt.imshow(img)

    return img


def load_vgg_model():
    model = models.vgg19(pretrained=True)
    model.eval()

    if use_cuda():
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
            p.requires_grad = False

    return model


def image_to_vgg_input_tensor(img):
    preprocessed_img = transform.resize(img, (224, 224))
    preprocessed_img = np.float32(preprocessed_img.copy())
    preprocessed_img = preprocessed_img[:, :, ::-1]

    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    for i in range(3):
        preprocessed_img[:, :, i] =\
            preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] =\
            preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = np.ascontiguousarray(
        np.transpose(preprocessed_img, (2, 0, 1))
    )

    if use_cuda():
        preprocessed_img_tensor =\
            torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor =\
            torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)

    return Variable(preprocessed_img_tensor, requires_grad = False)


def vgg_input_assessment(input_tensor, vgg_model):
    with open("data/imagenet1000_clsid_to_human.pkl", "r") as fp:
        vgg_class = pickle.load(fp)

    outputs = torch.nn.Softmax()(vgg_model(input_tensor))
    outputs_np = outputs.data.cpu().numpy()
    sorted_args = np.argsort(outputs_np[0, :])[::-1]

    print "5 top classes identified by the model:"
    print "(class index) class description: model score"

    for index in sorted_args[:5]:
        print "({}) {}: {}".format(index, vgg_class[index], outputs[0, index])

    print

    if outputs_np[0, sorted_args[0]] < 0.5:
        print "*** Warning ***"
        print "top category score under 0.5, extracted explanation may not be accurate on not well defined class"
        print


def use_cuda():
    return torch.cuda.is_available()


def image_tensor_to_numpy(tensor):
    img = tensor.data.cpu().numpy()[0]
    img = np.transpose(img, (1, 2, 0))
    return img


def post_processing(proc_img_tensor, pert_img_tensor):
    proc_img_np = image_tensor_to_numpy(proc_img_tensor)
    pert_img_np = image_tensor_to_numpy(pert_img_tensor)

    # mean over image channels:
    proc = np.mean(proc_img_np, axis=2)
    pert = np.mean(pert_img_np, axis=2)

    # highlighting the differences:
    diff = (proc - pert) ** 6

    # remove the edges: artifacts due to padding may appear.
    h, w = np.shape(diff)
    diff[:int(0.1 * h), :] = 0
    diff[int(0.9 * h):, :] = 0
    diff[:, :int(0.1 * w)] = 0
    diff[:, int(0.9 * w):] = 0

    # dilate the important points left for visibility:
    square = np.ones((20, 20))
    diff = morphology.dilation(diff, square)

    return diff, proc_img_np, pert_img_np


def plot_results(
    processed_img, pert_img, diff,
    indicate_center_of_mass=False,
):
    proc = np.mean(processed_img, axis=2)
    pert = np.mean(pert_img, axis=2)
    loc = center_of_mass(diff[::-1, :])

    fig, (ax1, ax2, ax3) = plt.subplots(
        ncols=3, figsize=(15, 5),
    )
    fig.canvas.set_window_title("images")

    im1 = ax1.pcolormesh(proc[::-1, :])
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    ax1.set_aspect(1)
    ax1.set_title("processed image")

    im2 = ax2.pcolormesh(pert[::-1, :])
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    ax2.set_aspect(1)
    ax2.set_title("perturbated image")

    im3 = ax3.pcolormesh(diff[::-1, :], cmap='Greys')
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.set_aspect(1)
    ax3.set_title("differences")
    if indicate_center_of_mass:
        ax3.annotate("X: center of mass", loc)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()