
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 원본 Qwen2-VL 관련 클래스 및 함수 임포트 (실제 환경에 맞게 import 경로 수정)
# ---------------------------------------------------------------------
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLDecoderLayer,
    Qwen2VLAttention,
    Qwen2VLSdpaAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
class Qwen2VLAttentionMasked(Qwen2VLAttention):
    """
    Qwen2VLAttention 모듈을 상속받아, forward() 내에서
    지정된 레이어/헤드 (예: "l23_h8"이면 layer 23의 head 8)에 대해 출력값을 0으로 마스킹합니다.
    """
    def __init__(self, config, layer_idx, mask_dict):
        super().__init__(config, layer_idx)
        self.mask_dict = mask_dict

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        # 1. 선형 투영: q, k, v 계산
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE 적용 (내부 또는 외부에서 주입)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # 3. 캐시 업데이트 (존재하면)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 4. key/value 헤드 반복 (필요시)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5. attention score 계산 및 softmax
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if query_states.dtype == torch.float16:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 6. attention output 계산
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"attn_output has unexpected shape: {attn_output.size()}")

        # 7. 마스킹 적용: 현재 레이어(self.layer_idx)에 해당하는 head 중,
        #    mask_dict에 지정된 head("l{layer}_h{head}")에 대해 출력을 0으로 설정.
        layer_prefix = f"l{self.layer_idx}_h"
        # import pdb;pdb.set_trace()
        for key in self.mask_dict:
            if key.startswith(layer_prefix):
                try:
                    head_idx = int(key.split("_h")[1])
                except Exception:
                    continue
                # 해당 head의 모든 출력값을 0으로 만듦.
                attn_output[:, head_idx, :, :] = 0.0
        # import pdb;pdb.set_trace()
        # 8. 헤드 출력을 재배열 후 projection 적용
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# =============================================================================
# 2. Qwen2VLDecoderLayerMasked
# =============================================================================


# =============================================================================
# 3. Qwen2VLModelMasked
# =============================================================================
class Qwen2VLModelMasked(Qwen2VLModel):
    """
    Qwen2VLModel을 상속받아, 디코더의 모든 레이어를 마스킹된 버전(Qwen2VLDecoderLayerMasked)으로 대체한 모델입니다.
    이때 이미지 인코더(visual encoder)는 변경하지 않고 language model 부분만 마스킹합니다.
    """
    def __init__(self, config, mask_dict):
        super().__init__(config)
        new_layers = []
        for layer_idx, layer in enumerate(self.layers):
            masked_layer = Qwen2VLDecoderLayerMasked(config, layer_idx, mask_dict)
            masked_layer.load_state_dict(layer.state_dict())
            device = next(layer.parameters()).device
            masked_layer = masked_layer.to(device).half()  # 강제로 half() 적용
            new_layers.append(masked_layer)
        self.layers = nn.ModuleList(new_layers)


# =============================================================================
# 4. Qwen2VLForConditionalGenerationMasked
# =============================================================================
class Qwen2VLForConditionalGenerationMasked(Qwen2VLForConditionalGeneration):
    """
    Qwen2VLForConditionalGeneration을 상속받아 language model 부분(self.model)을
    마스킹된 버전(Qwen2VLModelMasked)으로 교체합니다.
    
    from_pretrained() 호출 시 mask_dict 인자를 전달하면, 해당 정보에 따라 디코더의 일부 head가 0으로 마스킹됩니다.
    이미지 인코더(visual encoder)는 변경하지 않습니다.
    """
    def __init__(self, config, mask_dict: Optional[dict] = None, **kwargs):
        super().__init__(config)
        if mask_dict is not None:
            for i, layer in enumerate(self.model.layers):
                masked_layer = Qwen2VLDecoderLayerMasked(config, i, mask_dict)
                masked_layer.load_state_dict(layer.state_dict())
                device = next(layer.parameters()).device
                masked_layer = masked_layer.to(device).half()  # half() 적용
                self.model.layers[i] = masked_layer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, mask_dict: Optional[dict] = None, **kwargs):
        """
        from_pretrained() 호출 시 mask_dict 인자를 추가로 받을 수 있도록 오버라이딩합니다.
        기존에 불러온 모델의 language model 부분(디코더 레이어)을 마스킹된 버전으로 변환합니다.
        """
        instance = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if mask_dict is not None:
            for i, layer in enumerate(instance.model.layers):
                masked_layer = Qwen2VLDecoderLayerMasked(instance.config, i, mask_dict)
                masked_layer.load_state_dict(layer.state_dict())
                device = next(layer.parameters()).device
                masked_layer = masked_layer.to(device).half()  # half() 적용
                instance.model.layers[i] = masked_layer
        return instance

class Qwen2VLSdpaAttentionMasked(Qwen2VLSdpaAttention):
    """
    Qwen2VLSdpaAttention을 상속받아, SDPA 방식의 attention 결과에 대해
    지정된 레이어/헤드(예: "l23_h8"이면 layer 23의 head 8)의 출력을 0으로 마스킹합니다.
    
    출력 텐서는 SDPA 방식에서 최종 shape (bsz, q_len, hidden_size)로 나오므로,
    내부에서 head 단위로 복원한 후 masking을 적용하고 다시 재구성합니다.
    """
    def __init__(self, config, layer_idx, mask_dict):
        super().__init__(config, layer_idx)
        self.mask_dict = mask_dict

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if output_attentions:
            # SDPA does not support output_attentions; fallback if needed.
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        
        bsz, q_len, _ = hidden_states.size()
        # 선형 투영 및 view 변환
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if position_embeddings is None:
            # 경고 메시지 후 RoPE embeddings 계산
            from transformers import logging
            logging.get_logger(__name__).warning(
                "Transitioning to externally computed position embeddings; using internal computation for now."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        # SDPA: scaled_dot_product_attention returns (bsz, q_len, hidden_size) after appropriate transposes.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 
        # SDPA output: (bsz, q_len, hidden_size)
        # 복원: head 수와 head_dim으로 reshape하여 masking 적용
        # attn_output_reshaped = attn_output.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        # 
        # import pdb;pdb.set_trace()
        layer_prefix = f"l{self.layer_idx}_h"
        for key in self.mask_dict:
            if key.startswith(layer_prefix):
                try:
                    head_idx = int(key.split("_h")[1])
                except Exception:
                    continue
                attn_output[:, :, head_idx, :] = 0.0  # 마스킹 적용
        # import pdb;pdb.set_trace()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        # 원래 shape으로 복원 후 projection 적용
        # attn_output_masked = attn_output_reshaped.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output_masked = self.o_proj(attn_output)
        # import pdb;pdb.set_trace()
        return attn_output_masked, None, past_key_value

# =============================================================================
# 2. Qwen2VLDecoderLayerMasked (eager 및 sdpa 모두 지원)
# =============================================================================
class Qwen2VLDecoderLayerMasked(Qwen2VLDecoderLayer):
    """
    Qwen2VLDecoderLayer를 상속받아 self-attention 모듈을 masking된 버전으로 교체합니다.
    이미지 인코더 부분은 그대로 두고, language model(디코더)에서만 masking을 적용합니다.
    
    config._attn_implementation 값이:
      - "eager"이면 Qwen2VLAttentionMasked (기존 구현)
      - "sdpa"이면 Qwen2VLSdpaAttentionMasked를 사용합니다.
    """
    def __init__(self, config, layer_idx, mask_dict):
        super().__init__(config, layer_idx)
        if config._attn_implementation == "eager":
            self.self_attn = Qwen2VLAttentionMasked(config, layer_idx, mask_dict)
        elif config._attn_implementation == "sdpa":
            self.self_attn = Qwen2VLSdpaAttentionMasked(config, layer_idx, mask_dict)
        else:
            print(
                f"Warning: Masking is not implemented for attention implementation '{config._attn_implementation}'. "
                "Default self-attention is used."
            )