from  PIL import Image
import torch
import numpy as np
from qwen_vl_utils import process_vision_info
from torchvision.transforms import Normalize, Compose, Resize
from torchvision.transforms.functional import InterpolationMode

open_clip_vit_b_16_preprocess = Compose(
    [
        Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
def data_collate(batch, processor):
    messages = []
    images = []
    gt_imgs = []
    texts = []
    tokenized_texts = []
    for line in batch:
        # CLIP 模型的输入
        img_tensor = line[0]
        # clip_img_tensor = open_clip_vit_b_16_preprocess(img_tensor)
        images.append(img_tensor.numpy().tolist())
        tokenized_texts_tensor = line[3]
        tokenized_texts.append(tokenized_texts_tensor.numpy().tolist())
        # 将 tensor 类型的图片转为 Image 类型作为 Qwen 的输入。
        img_tensor_resize = img_tensor.permute(1, 2, 0)
        numpy_array = img_tensor_resize.numpy()
        numpy_array = numpy_array.astype(np.uint8)
        image = Image.fromarray(numpy_array)
        # 训练目标
        gt_imgs.append(line[1].numpy().tolist())
        # 计数目标的文本描述
        text = line[2]
        message = [{
            "role" : "user",
            "content" : [
                {
                    "type": "text",
                    "text": text
                 },
                 {
                     "type": "image",
                     "image": image
                 }
            ]
        }]
        messages.append(message)
        # 自定义 qwen 的 text 图文模板
        texts.append(text+"<|vision_start|><|image_pad|><|vision_end|>")
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    return qwen_inputs, torch.tensor(images), torch.tensor(tokenized_texts),torch.tensor(gt_imgs)


        
