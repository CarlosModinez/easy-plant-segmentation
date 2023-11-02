from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_mask(img, mask, random_color=False):
    fig, ax = plt.subplots()
    ax.imshow(img)
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis('off')
    return fig

def get_predictor(image) -> SamPredictor:
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    return predictor

def get_predicted_mask(predictor, input_point, input_label):    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks, scores, logits
