import torch
from segment_anything import SamPredictor, sam_model_registry
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class SAMProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="path/to/sam_vit_h.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image_path):
        with Image.open(image_path) as img:
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
            self.predictor.set_image(img_tensor.squeeze(0).permute(1, 2, 0).numpy())
            return img_tensor.squeeze(0).permute(1, 2, 0)

    def get_masks(self, bboxes):
        masks = []
        for bbox in bboxes:
            input_box = torch.tensor(
                [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
                device=self.device
            ).unsqueeze(0)
            masks.append(self.predictor.predict(box=input_box)[0])
        return masks

    def overlay_masks(self, image_tensor, masks):
        overlay_image = image_tensor.clone()
        for mask in masks:
            overlay_image[mask[0] > 0.5] = torch.tensor([1, 0, 0])  # Red for masks
        return to_pil_image(overlay_image.permute(2, 0, 1))
