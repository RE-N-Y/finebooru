# VQVAE training with animefaces

[blog](https://re-n-y.github.io/devlog/rambling/vqgan/) | [research.log](https://x.com/sleenyre)

This repository contains the code for training various variants of VQVAE on the animefaces dataset. I've also included a tiny FSQ model weights on `checkpoint/fsq/model.safetensors` if anyone wants to try it out. Note that I will be updating and cleaning the codebase over the next few weeks.

For an overview of results and experiments, please refer to the blog posts and previous research.log entries.

1. Dataset - [animefaces 2M](https://app.activeloop.ai/reny/animefaces/firstdbf9474d461a19e9333c2fd19b46115348f)
2. Quantization variants
    1. VQVAE
    2. Lookup free quantization
    3. Finite Scalar quantization
3. Backbone variants
    1. CNN (ConvNext / MetaTransformers)
    2. ViT (Vision Transformers)
    3. Mamba (VIM, Mamba Vision, Bidirectional Mamba)

## TODOS

- [ ] Upload model and dataset to HF
- [ ] Add overlapped training for ViT
- [ ] Add JAX support

## Acknowledgements

1. Andrej Karpathy / Suraj Patil for Gumbel softmax and ViT-VQGAN implementatiosn
2. Lucidrains and Mentzer et al. (FSQ authors) for providing reference Pytorch / JAX implementations
3. Tencent's MAGVITv2 for providing reference LFQ implementation
4. Tri Dao's lab for providing easy hackable codebase for Mamba variants
5. mamba vision / vision mamba repos for providing initial implementation ideas for Mamba variants