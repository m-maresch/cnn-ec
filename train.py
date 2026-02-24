# The following code is based on the PyTorch tutorial related to transfer learning for computer vision.
# Source: https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
# - Original License: BSD
# - Original Author: Sasank Chilamkurthy
# Original Copyright (c) 2017-2022, Pytorch contributors
# See PYTORCH_TUTORIALS_LICENSE in the project root for full license text of the PyTorch tutorials code.
# The code has been extended and modified to use LoftNN.

import argparse
import logging
import os
import time

from contextlib import nullcontext
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler

import torchvision

from torchvision import datasets, models, transforms

import loftnn

from loftnn import DataParallel, HybridPipelineParallel, PipelineParallel
from loftnn.configuration import HybridPipelinePlanningAlgorithm
from loftnn.types import Device, Worker


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_level", type=str, default="WARNING")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parallelism
    parser.add_argument("--parallelism", type=str)
    # pipeline parallelism
    parser.add_argument("--num_microbatches", type=int, default=4)
    # hybrid pipeline parallelism
    parser.add_argument(
        "--planner",
        type=HybridPipelinePlanningAlgorithm,
        default=HybridPipelinePlanningAlgorithm.exact,
    )
    parser.add_argument("--compute_capacities", type=str)
    parser.add_argument("--batch_size_limits", type=str, default="[]")
    parser.add_argument("--split_points", type=str)
    parser.add_argument("--device_groups", type=str)
    parser.add_argument("--samples_allocated", type=str)
    parser.add_argument("--activation_checkpointing_budgets", type=str, default="[]")

    args = parser.parse_args()

    device = args.device
    log_level = args.log_level
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    parallelism = args.parallelism
    num_microbatches = args.num_microbatches

    def eval_complex_arg(arg):
        if arg is None:
            return None
        return eval(arg)

    planner = args.planner
    compute_capacities = eval_complex_arg(args.compute_capacities)
    batch_size_limits = eval_complex_arg(args.batch_size_limits)
    split_points = eval_complex_arg(args.split_points)
    device_groups = eval_complex_arg(args.device_groups)
    samples_allocated = eval_complex_arg(args.samples_allocated)
    activation_checkpointing_budgets = eval_complex_arg(
        args.activation_checkpointing_budgets
    )

    return (
        device,
        log_level,
        num_epochs,
        batch_size,
        gradient_accumulation_steps,
        parallelism,
        num_microbatches,
        planner,
        compute_capacities,
        batch_size_limits,
        split_points,
        device_groups,
        samples_allocated,
        activation_checkpointing_budgets,
    )


(
    device,
    log_level,
    num_epochs,
    batch_size,
    gradient_accumulation_steps,
    parallelism,
    num_microbatches,
    planner,
    compute_capacities,
    batch_size_limits,
    split_points,
    device_groups,
    samples_allocated,
    activation_checkpointing_budgets,
) = parse_args()

logging.basicConfig(level=log_level)

is_data_parallel = parallelism == "data"
is_pipeline_parallel = parallelism == "pipeline"
is_hybrid_pipeline_parallel = parallelism == "hybrid"

has_pipeline_parallelism = is_pipeline_parallel or is_hybrid_pipeline_parallel

print(
    f"""Using:
    device = {device},
    log_level = {log_level},
    num_epochs = {num_epochs},
    batch_size = {batch_size},
    gradient_accumulation_steps = {gradient_accumulation_steps},
    parallelism = {parallelism}"""
)

if is_pipeline_parallel:
    print(
        f"""With:
    num_microbatches = {num_microbatches},
    split_points = {split_points}
    """
    )
elif is_hybrid_pipeline_parallel:
    print(
        f"""With:
    num_microbatches = {num_microbatches},
    planner = {planner},
    compute_capacities = {compute_capacities},
    batch_size_limits = {batch_size_limits},
    split_points = {split_points},
    device_groups = {device_groups},
    samples_allocated = {samples_allocated}
    activation_checkpointing_budgets = {activation_checkpointing_budgets}
    """
    )
else:
    print()

if loftnn.is_available():
    print("LoftNN is available")
    process_config = loftnn.ProcessConfiguration.from_env()
    master_process = process_config.rank == 0

    seed_offset = 0
    if is_data_parallel:
        seed_offset = process_config.rank  # each process gets a different seed

    if gradient_accumulation_steps > 1:
        assert (
            not has_pipeline_parallelism
        ), "misconfiguration: gradient accumulation used with pipeline parallelism"
        assert gradient_accumulation_steps % process_config.world_size == 0
        gradient_accumulation_steps //= process_config.world_size
else:
    print("LoftNN is not available")
    process_config = loftnn.ProcessConfiguration(rank=0, local_rank=0, world_size=1)
    master_process = True
    seed_offset = 0

no_split_or_last_process = (
    not has_pipeline_parallelism or process_config.rank == process_config.world_size - 1
)

master_or_last_process = (master_process and not has_pipeline_parallelism) or (
    has_pipeline_parallelism and process_config.rank == process_config.world_size - 1
)


def seed():
    torch.manual_seed(1445 + seed_offset)


seed()

cudnn.benchmark = True

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "data/hymenoptera_data"
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders["train"]))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

criterion = nn.CrossEntropyLoss()
corrects = []


def loss_fn(outputs, labels):
    outputs = outputs.to(device)
    loss = criterion(outputs, labels)

    global corrects
    _, preds = torch.max(outputs, 1)
    corrects.append((torch.sum(preds == labels.data), len(preds)))

    return loss


def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        model_params_path = (
            os.path.join(tempdir, f"model_params_rank_{process_config.rank}.pt")
            if has_pipeline_parallelism
            else os.path.join(tempdir, "best_model_params.pt")
        )

        torch.save(model.state_dict(), model_params_path)
        best_acc = 0.0

        epoch_times = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                    epoch_train_start = time.time()
                else:
                    model.eval()  # Set model to evaluate mode

                num_iters = len(dataloaders[phase])

                global corrects
                corrects.clear()
                losses = torch.zeros(num_iters - 1)

                idx = 1
                batch_iter = iter(dataloaders[phase])
                inputs, labels = next(batch_iter)

                done = False
                while not done:
                    for micro_step in range(gradient_accumulation_steps):
                        if idx == num_iters - 1:
                            done = True
                            break

                        if is_data_parallel:
                            model.require_backward_grad_sync = (
                                micro_step == gradient_accumulation_steps - 1
                            )

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # forward + backward
                        # track history if only in train
                        with (
                            nullcontext()
                            if has_pipeline_parallelism
                            else torch.set_grad_enabled(phase == "train")
                        ):
                            if has_pipeline_parallelism:
                                outputs, loss = model(inputs, labels)
                            else:
                                outputs = model(inputs)
                                loss = loss_fn(outputs, labels)
                                if phase == "train":
                                    loss.backward()

                        if no_split_or_last_process:
                            loss = (
                                loss / gradient_accumulation_steps
                            )  # scale the loss to account for gradient accumulation

                        # statistics
                        if master_or_last_process:
                            losses[idx] = loss.item() * gradient_accumulation_steps

                        idx += 1
                        inputs, labels = next(batch_iter)

                    # optimize only if in training phase
                    if phase == "train":
                        optimizer.step()
                        # flush the gradients as soon as we can, no need for this memory anymore
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                epoch_time = None
                if phase == "train":
                    epoch_train_end = time.time()
                    epoch_time = epoch_train_end - epoch_train_start
                    epoch_times.append(epoch_time)

                if master_or_last_process:
                    epoch_loss = losses.mean()
                    epoch_acc = sum(
                        [correct_preds.item() for (correct_preds, _) in corrects]
                    ) / sum([num_preds for (_, num_preds) in corrects])

                    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    if epoch_time is not None:
                        print(f"{phase} Time: {epoch_time*1000:.2f}ms")

                    # deep copy the model
                    if phase == "val" and epoch_acc > best_acc:
                        best_acc = epoch_acc

                        if not has_pipeline_parallelism:
                            torch.save(model.state_dict(), model_params_path)

                if phase == "val" and has_pipeline_parallelism:
                    torch.save(model.state_dict(), model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Epoch times: {epoch_times}s")

        if master_or_last_process:
            print(f"Best val Acc: {best_acc:4f}")

    return model


model_ft = models.resnet50(weights="IMAGENET1K_V1")
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
inputs = inputs.to(device)

if is_data_parallel:
    dist_model = DataParallel(
        model=model_ft,
        process_config=process_config,
        device=Device.cuda if device == "cuda" else Device.cpu,
    )
elif is_pipeline_parallel:
    microbatch_sample = inputs.chunk(num_microbatches)[0]
    pipeline_config = loftnn.PipelineConfiguration(
        split_points=split_points,
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = PipelineParallel(
        model_ft,
        process_config=process_config,
        pipeline_config=pipeline_config,
    )
elif is_hybrid_pipeline_parallel:
    microbatch_sample = inputs.chunk(num_microbatches)[0]
    pipeline_config = loftnn.HybridPipelineConfiguration(
        planner=planner,
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = HybridPipelineParallel(
        model_ft,
        process_config=process_config,
        hybrid_pipeline_config=pipeline_config,
        device=Device.cuda if device == "cuda" else Device.cpu,
    )

    if split_points and device_groups and samples_allocated:
        device_groups_complete = [0] + device_groups + [process_config.world_size]
        samples_allocated = [
            {
                Worker(
                    rank=r,
                    compute_capacity=(
                        compute_capacities[r] if compute_capacities else 1 / 1000
                    ),
                    batch_size_limits=(
                        batch_size_limits[r] if batch_size_limits else 100
                    ),
                ): samples_allocated[r]
                for r in range(device_groups_complete[g], device_groups_complete[g + 1])
            }
            for g in range(len(device_groups_complete) - 1)
        ]
    else:
        (
            split_points,
            device_groups,
            samples_allocated,
            activation_checkpointing_budgets,
        ) = dist_model.compute_plan(batch_size_limits)

    print("using hybrid pipeline parallelism with the following plan:")
    print(f"    split points = {split_points}")
    print(f"    device groups = {device_groups}")
    print(f"    samples allocated = {samples_allocated}")
    print(f"    activation checkpointing budgets = {activation_checkpointing_budgets}")

    dist_model.prepare_schedule(
        split_points, device_groups, samples_allocated, activation_checkpointing_budgets
    )

    # computing the plan advances the RNG of the master process
    seed()  # needed to make sure that X and Y are aligned during training

if parallelism is not None:
    model_ft = dist_model

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)


def visualize_model(model, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train()
                    return
        model.train()


def visualize_model_predictions(model, img_path):
    model.eval()

    img = Image.open(img_path)
    img = data_transforms["val"](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)

        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2, 2, 1)
        ax.axis("off")
        ax.set_title(f"Predicted: {class_names[preds[0]]}")
        imshow(img.cpu().data[0])

        model.train()


if parallelism is None:
    visualize_model(model_ft)

    plt.show()

    visualize_model_predictions(
        model_ft, img_path="data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg"
    )

    plt.show()
