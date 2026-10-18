"""Background removal for images without transparency.

The original TRELLIS.2 uses RMBG-2.0 (BRIA: sign-up required, non-commercial).
RMBG-2.0 is a BiRefNet model; this loads the original BiRefNet weights (MIT,
no sign-up) into TRELLIS.2's own BiRefNet wrapper, so only the weights differ.
The model's code comes from its Hugging Face repository, pinned to the revision
checked here.
"""
import torch

REPO = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def birefnet():
    """Returns TRELLIS.2's BiRefNet wrapper holding the MIT weights (on the CPU)."""
    from transformers import AutoModelForImageSegmentation
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet
    wrapper = BiRefNet.__new__(BiRefNet)
    model = AutoModelForImageSegmentation.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    # The weights are stored in float16; the wrapper feeds float32 images.
    wrapper.model = model.float().eval()
    from torchvision import transforms
    wrapper.transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return wrapper


@torch.no_grad()
def remove_background(image, device):
    """RGB image -> RGBA with the subject's mask as alpha (for saving or checking)."""
    model = birefnet().to(device)
    try:
        return model(image.convert("RGB"))
    finally:
        del model
