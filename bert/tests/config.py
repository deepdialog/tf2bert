params = {
    "attention_probs_dropout_prob": 0.1,
    # "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pooler_fc_size": 768,
    # "pooler_num_attention_heads": 12,
    # "pooler_num_fc_layers": 3,
    # "pooler_size_per_head": 128,
    # "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 21128
}

albert_params = {
    "attention_probs_dropout_prob": 0,
    "hidden_act": "relu",
    "hidden_dropout_prob": 0,
    "embedding_size": 128,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    # "num_hidden_groups": 1,
    # "net_structure_type": 0,
    # "layers_to_keep": [],
    # "gap_size": 0,
    # "num_memory_blocks": 0,
    # "inner_group_num": 1,
    # "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 21128
}

# For CI testing only
params_small = {
    "attention_probs_dropout_prob": 0.1,
    # "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128 * 4,
    "max_position_embeddings": 64,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "pooler_fc_size": 64,
    # "pooler_num_attention_heads": 12,
    # "pooler_num_fc_layers": 3,
    # "pooler_size_per_head": 128,
    # "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    "vocab_size": 64
}
