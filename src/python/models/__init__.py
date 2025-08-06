# models/__init__.py
# This package will contain all model definitions.
def get_model(config):
    model_name = config.get('model.name', 'unet').lower()
    if model_name == 'unet':
        from .unet import UNet
        return UNet(config)
    elif model_name == 'pmnet_v1':
        from .pmnet_v1 import PMNet as PMNetV1
        return PMNetV1(config)
    elif model_name == 'pmnet_v3':
        from .pmnet_v3 import PMNet as PMNetV3
        return PMNetV3(config)
    elif model_name == 'radionet':
        from .radionet import RadioWNet 
        return RadioWNet(config)
    elif model_name == 'vgg16':
        from .vgg16 import Model as VGG16
        return VGG16(config)
    elif model_name == 'transunet':
        from .transunet import VisionTransformer as TransUNet
        from .transunet import CONFIGS
        import numpy as np
        net =  TransUNet(config=CONFIGS['ViT-SMOL'])
        return net
    elif model_name == 'tiny_vit':
        from .tiny_vit import TinyViT, tiny_vit_5m_256
        import numpy as np
        net = tiny_vit_5m_256()
        # net.load_from(weights=np.load("src/python/models/pretrained/tiny_vit_5m_22kto1k_distill.pth"))
        return net
    # Add more models here as needed
    else:
        raise ValueError(f"Unknown model name: {model_name}")
