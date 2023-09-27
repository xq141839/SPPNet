import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.mask_decoder import MaskDecoder
from modeling.prompt_encoder import PromptEncoder
from modeling.transformer import TwoWayTransformer
from modeling.tiny_vit_sam import TinyViT
from utils.transforms import ResizeLongestSide
from modeling.image_encoder import ImageEncoderViT
from functools import partial

class LLSIE(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super(LLSIE, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=  kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x

class Model(nn.Module):
    def __init__(self, image_encoder):
        super(Model, self).__init__()
        self.image_encoder = image_encoder
        # self.image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        #     embed_dims=[64, 128, 160, 320],
        #     depths=[2, 2, 6, 2],
        #     num_heads=[2, 4, 5, 10],
        #     window_sizes=[7, 7, 14, 7],
        #     mlp_ratio=4.,
        #     drop_rate=0.,
        #     drop_path_rate=0.0,
        #     use_checkpoint=True,
        #     mbconv_expand_ratio=4.0,
        #     local_conv_size=3,
        #     layer_lr_decay=0.8
        # )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64), # 1024 // 16
            input_image_size=(1024, 1024),
            mask_in_chans=16,
            )
        self.mask_decoder =  MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            ) 
        self.transform = ResizeLongestSide(1024)

        self.conv1 = LLSIE(3, 32)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x_resized, point_coords, point_labels, x, img_shape):

        low_level_infos = self.conv1(x_resized)

        image_embeddings = self.image_encoder(x)

        transformed_coords = self.transform.apply_coords_torch(point_coords, img_shape)

        outputs = []
        
        for one_coords, one_label, one_x, lli in zip(transformed_coords, point_labels, image_embeddings, low_level_infos):
        # for one_coords, one_label, one_x in zip(transformed_coords, point_labels, image_embeddings):
            
            one_coords = one_coords.unsqueeze(0)
            one_label = one_label.unsqueeze(0)

            points = (one_coords, one_label)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
        
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=one_x.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                low_level_info=lli,
            )
            
            outputs.append(low_res_masks.squeeze(0))

        return torch.stack(outputs, dim=0)
