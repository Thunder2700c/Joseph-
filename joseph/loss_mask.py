"""Assistant-response target masks for mixed language-model and SFT corpora."""

from __future__ import annotations


def build_loss_mask(
    sample_ids: list[int],
    *,
    assistant_id: int,
    eot_id: int,
) -> list[int]:
    """Mask user/prompt tokens and train only assistant turns in SFT samples.

    Ordinary prose samples contain no assistant marker and retain full causal-LM
    loss. Dialogue samples must contain balanced assistant-marker / EOT pairs.
    The returned mask aligns one-to-one with [BOS] + sample + [EOS].
    """
    is_dialogue = assistant_id in sample_ids
    if not is_dialogue:
        return [0] + [1] * len(sample_ids) + [1]

    mask = [0]  # BOS is input context, never a prediction target.
    in_assistant_turn = False
    assistant_turns = 0
    for token_id in sample_ids:
        if token_id == assistant_id:
            if in_assistant_turn:
                raise ValueError("nested assistant marker before end-of-turn")
            in_assistant_turn = True
            assistant_turns += 1
            mask.append(1)  # Learn to begin the assistant turn.
        elif token_id == eot_id:
            if not in_assistant_turn:
                raise ValueError("end-of-turn outside an assistant response")
            mask.append(1)  # Learn when to stop the response.
            in_assistant_turn = False
        else:
            mask.append(int(in_assistant_turn))

    if in_assistant_turn:
        raise ValueError("dialogue sample has an unterminated assistant response")
    if assistant_turns == 0:
        raise ValueError("dialogue sample has no assistant response")
    mask.append(1)  # Learn EOS after the final EOT.
    if len(mask) != len(sample_ids) + 2:
        raise AssertionError("loss-mask/token alignment failure")
    return mask
