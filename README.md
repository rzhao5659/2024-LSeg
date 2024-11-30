## ViT changes

To make ViT work with arbitrary image size as input, make these changes during inference for each input image x:

1. Change attribute `image_size` (assumes H = W = `image_size`) of the ViT model. This is necessary to bypass the assert check `self.image_size == h == w` in `VisionTransformer._process_input(x)`.
2. Interpolate the learned position embedding by using their helper function `interpolate_embeddings` in `torchvision/models/vision_transformer.py`
   Link: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L268

## DPT for LSeg

To use DPT as an pixel-wise encoder. Initialize it with this, where `C` is the desired multimodal embedding dimension. \
`model = DPT(head=nn.Identity(), output_feature_dim=C)`
