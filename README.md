# CNN and ViT training with LoftNN

This repository adapts [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) to integrate the parallelism types provided by [LoftNN](https://github.com/m-maresch/loftnn) for distributed training.

Support for both CNNs (ResNets) and ViTs (Swin Transformers) is provided.

## Example usage
```
python train.py --device=cuda --parallelism='data' --log_level="INFO"

python train.py --device=cuda --parallelism='pipeline' --log_level="INFO" --split_points="[54,106]"

python train.py --device=cuda --parallelism='hybrid' --log_level="INFO" --planner="ecs"

python train.py --device=cuda --parallelism='hybrid' --log_level="INFO" --planner="exact"
```

Note that multiple environment variables need to be set for LoftNN to be available for distributed training. See details [here](https://github.com/m-maresch/loftnn#distributed-training).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. Third-party library notices are documented in `THIRD-PARTY-NOTICES.txt`.

The code is based on [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), which is licensed under the BSD License. See the code for details.

## Dependencies

Thanks to everyone contributing to any of the following projects and tutorials:

- PyTorch, torchvision, and the [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- NumPy
- Matplotlib
- Pillow
