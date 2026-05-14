from __future__ import annotations

from types import SimpleNamespace

import torch

from Coconut.coconut import Coconut
from lsp_jepa.adapters.coconut_lsp_adapter import CoconutLSPAdapter


class TinyCacheLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.calls = 0
        with torch.no_grad():
            weights = torch.arange(
                vocab_size * hidden_size,
                dtype=torch.float32,
            ).reshape(vocab_size, hidden_size)
            self.embedding.weight.copy_(weights)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
    ):
        del attention_mask, position_ids, output_hidden_states
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]

        self.calls += 1
        hidden_states = inputs_embeds + float(self.calls * 10)
        logits = inputs_embeds.new_zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.vocab_size,
        )
        total_len = past_len + inputs_embeds.shape[1]
        cache = inputs_embeds.new_zeros(inputs_embeds.shape[0], 1, total_len, 1)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden_states,),
            past_key_values=((cache, cache.clone()),),
        )


def test_coconut_forward_returns_post_latent_states_not_latent_inputs():
    latent_id = 2
    start_id = 3
    end_id = 4
    eos_id = 5
    base = TinyCacheLM()
    model = Coconut(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=eos_id,
    )
    input_ids = torch.tensor([[11, start_id, latent_id, latent_id, end_id, eos_id]])
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).reshape(1, -1)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )

    previous_visible_state = base.embedding(input_ids)[0, 1, :]
    expected_h0 = previous_visible_state + 10.0
    expected_h1 = expected_h0 + 20.0
    expected_h2 = expected_h1 + 30.0

    assert outputs.latent_mask.tolist() == [[True, True]]
    assert outputs.latent_states.shape == (1, 2, base.embedding.embedding_dim)
    assert torch.allclose(outputs.inputs_embeds[0, 2, :], expected_h0)
    assert torch.allclose(outputs.inputs_embeds[0, 3, :], expected_h1)
    assert torch.allclose(outputs.latent_states[0, 0, :], expected_h1)
    assert torch.allclose(outputs.latent_states[0, 1, :], expected_h2)


def test_coconut_adapter_prefers_post_latent_host_states():
    latent_id = 2
    start_id = 3
    end_id = 4
    eos_id = 5
    base = TinyCacheLM()
    model = Coconut(
        base,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=eos_id,
    )
    input_ids = torch.tensor(
        [
            [11, start_id, latent_id, latent_id, end_id, eos_id],
            [12, start_id, latent_id, end_id, eos_id, 0],
        ]
    )
    labels = input_ids.clone()
    labels[1, -1] = -100

    output = CoconutLSPAdapter(latent_token_id=latent_id).forward_student(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0],
                ]
            ),
            "labels": labels,
            "position_ids": torch.arange(input_ids.shape[1]).repeat(2, 1),
        },
        output_latent_states=True,
    )

    assert output.debug["latent_source"] == "host_output.latent_states"
    assert output.latent_mask.tolist() == [[True, True], [True, False]]
    assert output.latent_states.shape == (2, 2, base.embedding.embedding_dim)
    assert torch.equal(
        output.latent_states[1, 1, :],
        torch.zeros_like(output.latent_states[1, 1, :]),
    )
