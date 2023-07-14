import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.mask_decoder import MaskDecoder
from modeling.prompt_encoder import PromptEncoder
from modeling.transformer import TwoWayTransformer
from modeling.tiny_vit_sam import TinyViT

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
    
    def forward(self, point_coords, point_labels, image_embeddings):

        # image_embeddings = self.image_encoder(x)

        points = (point_coords, point_labels)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
       
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # print(low_res_masks.shape)

            # masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            # masks = masks > self.mask_threshold
            # outputs.append(
            #     {
            #         "masks": masks,
            #         "iou_predictions": iou_predictions,
            #         "low_res_logits": low_res_masks,
            #     }
            # )

        return low_res_masks
