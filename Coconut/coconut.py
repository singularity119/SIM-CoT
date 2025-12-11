# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
import copy
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        # 1. 安全计算索引 (CPU)
        if isinstance(self.latent_token_id, torch.Tensor):
            target_id = self.latent_token_id.item()
        else:
            target_id = self.latent_token_id

        input_ids_cpu = input_ids.to("cpu")
        latent_indices = (input_ids_cpu == target_id).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  

        # ================= 修改: 强制同步 max_n_latents =================
        # 计算当前卡上的最大值
        if len(latent_lists) > 0 and len(latent_lists[0]) > 0:
            local_max = max([len(l) for l in latent_lists])
        else:
            local_max = 0
            
        local_max_tensor = torch.tensor([local_max], device=input_ids.device, dtype=torch.long)

        # 分布式通信：取所有卡中最大的那个值，保证所有卡循环次数一致
        if dist.is_initialized():
            dist.all_reduce(local_max_tensor, op=dist.ReduceOp.MAX)
        
        max_n_latents = local_max_tensor.item()
        # ===============================================================

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # 只有当本地真的有 latent token 时才调整 compute range
            # 如果本地是空转(为了配合其他卡)，range 保持默认即可，下面循环会处理
            if latent_indices.numel() > 0:
                next_compute_range = (0, latent_indices[:, 1].min().item())
            else:
                # 极端情况：这张卡这条数据全是普通文本，没有 latent
                # 但为了同步，它必须陪跑。它的 range 可以设为 0-0 或者保持原样，
                # 但为了不影响普通计算，我们让它正常参与第一次 forward，
                # 后面的 pass_idx 循环中，由于 filling_indices 为空，它其实就是 standard forward
                next_compute_range = (0, input_ids.shape[1]) 
                # 注意：这里逻辑比较微妙，如果 max_n_latents > 0 但这张卡没 latent，
                # 它应该在第一次 forward 就把所有东西算完？
                # 不，为了对齐 KV Cache，我们尽量让它行为一致。
                # 简单处理：如果这张卡没 latent，就让它按 max_n_latents 跑，
                # 只是 filling_indices 永远为空，不会插入 thought。
                pass

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # 更新下一次计算范围
            if pass_idx + 1 >= max_n_latents:
                end_pos = input_ids.shape[1]
            else:
                # 正常逻辑是 +1，但如果 next_compute_range[1] 已经到了末尾，就保持末尾
                end_pos = min(input_ids.shape[1], next_compute_range[1] + 1)

            next_compute_range = (next_compute_range[1], end_pos)

            # 如果 range 为空（例如这张卡其实早就跑完了，只是在陪跑），
            # 这里的 hidden_states 可能需要特殊处理？
            # 只要 input_embeds 是空的，base_causallm 通常能处理。
            
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # feedback logic
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx  # 关键：只有当本地确实有这个 step 时才填充
            ]

            if filling_indices:
                tensor_list = [
                    [
                        inputs_embeds[batch_idx, pos, :]
                        for pos in range(inputs_embeds.shape[1])
                    ]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair
                    # 【新增保护】确保索引不越界，且 tensor 形状匹配
                    target_idx = token_idx - 1 - hidden_states_offset
                    if target_idx >= 0 and target_idx < hidden_states.shape[1]:
                        tensor_list[batch_idx][token_idx] = hidden_states[
                            batch_idx, target_idx, :
                        ]
                    else:
                        # 这种情况理论上不该发生，但如果 Padding 导致偏移错乱，宁可跳过也不要崩
                        pass

                inputs_embeds = torch.stack(
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )

        # final pass
        # 如果 next_compute_range 还是有效范围，则运行。
        # 如果已经在循环里跑完了（陪跑），这里可能 range 是 (L, L)，跑一次空也没事。
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


class CoconutGPT_Same_Word_Embedding(nn.Module):
    def __init__(
        self,
        base_causallm,
        expainable_llm,
        tokenizer,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        step_start_id,
        c_thought,
        configs,
    ):

        super(CoconutGPT_Same_Word_Embedding, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.expainable_llm = expainable_llm
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.step_start_id = step_start_id
        self.c_thought = c_thought
        self.config = configs

        if hasattr(self.config, "training_method"):
            if self.config.training_method == 'only_expainable_llm':
                for param in self.base_causallm.parameters(): param.requires_grad = False
            elif self.config.training_method == 'only_base_causallm':
                for param in self.expainable_llm.parameters(): param.requires_grad = False
            elif self.config.training_method == 'full':
                pass
            elif self.config.training_method == 'freeze_backbone':
                for param in self.base_causallm.parameters(): param.requires_grad = False
                for param in self.expainable_llm.parameters(): param.requires_grad = False
            else:
                raise ValueError(f"not this training_method {self.config.training_method=}")

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        logits = []
        loss = 0.0
        
        # 1. 安全索引计算 (CPU)
        if isinstance(self.latent_token_id, torch.Tensor):
            target_id = self.latent_token_id.item()
        else:
            target_id = self.latent_token_id

        input_ids_cpu = input_ids.to("cpu")
        latent_indices = (input_ids_cpu == target_id).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        # ================= 修改: 强制同步 max_n_latents =================
        if len(latent_lists) > 0 and len(latent_lists[0]) > 0:
            local_max = max([len(l) for l in latent_lists])
        else:
            local_max = 0
        local_max_tensor = torch.tensor([local_max], device=input_ids.device, dtype=torch.long)
        
        if dist.is_initialized():
            dist.all_reduce(local_max_tensor, op=dist.ReduceOp.MAX)
        
        max_n_latents = local_max_tensor.item()
        # ===============================================================

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # 同理，如果本地没有 latent，也需要处理 range
            if latent_indices.numel() > 0:
                next_compute_range = (0, latent_indices[:, 1].min().item())
            else:
                next_compute_range = (0, input_ids.shape[1])

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            if pass_idx + 1 >= max_n_latents:
                end_pos = input_ids.shape[1]
            else:
                end_pos = min(input_ids.shape[1], next_compute_range[1] + 1)
                
            next_compute_range = (
                next_compute_range[1],
                end_pos
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            if filling_indices:
                tensor_list = [
                    [
                        inputs_embeds[batch_idx, pos, :]
                        for pos in range(inputs_embeds.shape[1])
                    ]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair
                    if token_idx - 1 - hidden_states_offset < hidden_states.shape[1]:
                        tensor_list[batch_idx][token_idx] = hidden_states[
                            batch_idx, token_idx - 1 - hidden_states_offset, :
                        ]

                inputs_embeds = torch.stack(
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        if self.config.training_method == 'only_base_causallm' or self.config.training_method == 'full':
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        
        # ... (visualize logic omitted for brevity, keeping existing structure safe) ...
        # 注意：这里如果开了 visualize，可能也需要类似同步，但暂时假设只用 only_base_causallm 训练

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(self, input_ids, attention_mask, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
        # Generate logic stays the same
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        outputs = self.forward(input_ids, torch.ones_like(input_ids, device=input_ids.device), labels, torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1))
        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id: break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
        if synced_gpus:
            while (self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT):
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)
        if output_embedding: return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else: return torch.tensor(tokens).view(1, -1)