from src.model_pretrain import *

N_top = 5


def inferenceNetMedGpt(input, node_type, mask_index_question, nodes, edge, model):
    ##########################
    device = torch.device('cpu')

    mask_token = edge['z_index'].max() + 1

    nodes_at_mask = nodes.loc[nodes['node_type'] == node_type, ['node_index', 'node_name']].values
    node_ids_at_mask = torch.tensor(nodes_at_mask[:, 0].astype(int)).to(device)

    input = [int(i) for i in input.split(',')]
    sentence = torch.tensor([input])
    # mask_pos = int(mask_index_question)  # absolute position of the MASK

    # Treat --mask_index_question as an ABSOLUTE token position
    mask_pos = int(mask_index_question)

    # Safety check
    seq_len = sentence.size(1)
    if not (0 <= mask_pos < seq_len):
        raise IndexError(
            f"--mask_index_question {mask_pos} out of bounds; sequence length={seq_len}")

    if sentence[0, mask_pos].item() != mask_token:
        mask_positions = torch.where(sentence == mask_token)[1].tolist()
        raise ValueError(
            f"Token at position {mask_pos} is not MASK "
            f"(got {sentence[0, mask_pos].item()}, expected {mask_token}). "
            f"MASK positions in this sentence: {mask_positions}"
        )

    input = sentence.to(device)
    with torch.no_grad():
        output = model(input)
        logits = output[:, mask_pos, node_ids_at_mask]
        probs = F.softmax(logits, dim=1)

        top_probs, top_idx = torch.topk(probs, N_top,
                                        dim=1)  # top_idx are positions in node_ids_at_mask
        top_idx = top_idx.cpu().numpy()[0]  # shape (N_top,)

        drug_names = nodes_at_mask[top_idx, 1]

    del sentence, output, logits, probs, top_probs, top_idx, node_ids_at_mask

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return drug_names
