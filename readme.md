# DDPM Implementation

This project is an implementation of the paper:  
**Denoising Diffusion Probabilistic Models (DDPM)**  
📄 [Original Paper](https://arxiv.org/abs/2006.11239) by Jonathan Ho et al.

## Dataset

- **CelebA** dataset was used for training.
- A subset of **5000 images** was used.
- The model was trained for **300 epochs** with **700 diffusion timesteps**.

## Getting Started

### 🔄 Clone the Repository

```bash
git clone https://github.com/arnabroy734/ddpm.git
cd ddpm
```

### 🛠️ Training the Model

1. All training configurations are available in the `config.yaml` file.
2. Set the GPU to use (if multiple GPUs are available):

```bash
export CUDA_VISIBLE_DEVICES=0  # Change '0' to the desired GPU ID
```

3. Start training:

```bash
python train.py
```

### 🧪 Inference

To generate images using the trained model, follow the instructions in the Jupyter notebook:

```bash
generation.ipynb
```

It demonstrates how to load the model and sample new images.



## License

This project is open-source and available under the MIT License.

## Contact

For any queries, feel free to open an issue or reach out to the repository owner.
