# hook_siglip.py

import torch

class SigLIPHookLogger:
    def __init__(self, model, device="cuda:0", layer_index=23):
        self.model = model
        self.device = device
        self.layer_index = layer_index
        self.attentions = []  # [B, N, D]
        self.mlps = []        # [B, N, D]
        self._register_hooks()

    def _register_hooks(self):
        def hook_attention(module, input, output):
            self.attentions.append(output.detach())

        def hook_mlp(module, input, output):
            self.mlps.append(output.detach())

        self.model.vision_model.encoder.layers[self.layer_index].self_attn.out_proj.register_forward_hook(hook_attention)
        self.model.vision_model.encoder.layers[self.layer_index].mlp.fc2.register_forward_hook(hook_mlp)

    def reinit(self):
        self.attentions = []
        self.mlps = []
        torch.cuda.empty_cache()

    def finalize(self, normalize_target=None):
        """
        返回: attention_map, mlp_map
        - attention_map: [B, H, W]
        - mlp_map:       [B, H, W]
        """
        assert len(self.attentions) > 0 and len(self.mlps) > 0, "You must run model forward first."

        attn = self.attentions[0]  # shape [B, N, D]
        mlp = self.mlps[0]         # shape [B, N, D]

        # 通常 SigLIP 的 patch 数是 576 = 24 x 24
        H = W = int(attn.shape[1] ** 0.5)

        if normalize_target is not None:
            normalize_target = torch.nn.functional.normalize(normalize_target, dim=-1)
            attn_map = torch.einsum("bnd,bd->bn", attn, normalize_target).view(attn.size(0), H, W)
            mlp_map = torch.einsum("bnd,bd->bn", mlp, normalize_target).view(mlp.size(0), H, W)
            return attn_map, mlp_map
        else:
            return attn.view(attn.size(0), H, W, -1), mlp.view(mlp.size(0), H, W, -1)
