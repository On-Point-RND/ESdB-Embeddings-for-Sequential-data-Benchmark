import torch



class TransformerBlockFast(torch.nn.Module):
    """
    TransformerBlock with custom layers
    """

    def __init__(
        self,
        hidden_size: int,
        attn_heads: int,
        feed_forward_hidden: int,
        dropout: float,
        acceleration_config: dict | None = None,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param attn_heads: Head sizes of multi-head attention.
        :param feed_forward_hidden: Feed_forward_hidden, usually 4*hidden_size.
        :param dropout: Dropout rate.
        :acceleration_config: Parameters for acceleration.
        """
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            hidden_size, attn_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.attention_norm = torch.nn.LayerNorm(hidden_size)

        acceleration_config = acceleration_config or {}
        if acceleration_config.get("pff_block"):  # TODO: Here we should add some checks
            self.pff = PositionwiseFeedForwardFast(
                d_model=hidden_size,
                d_ff=feed_forward_hidden,
                dropout=dropout,
                acceleration_config=acceleration_config["pff_block"],
            )
        else:
            self.pff = PositionwiseFeedForward(
                d_model=hidden_size, d_ff=feed_forward_hidden, dropout=dropout
            )

        self.pff_dropout = torch.nn.Dropout(dropout)
        self.pff_norm = torch.nn.LayerNorm(hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: Input bert embedding.
        :param mask: Mask where 0 - <MASK>, 1 - otherwise.

        :returns: Embedding after Transformer block applied.
        """
        # Attention + skip-connection
        x_norm = self.attention_norm(x)
        attent_emb, _ = self.attention(
            x_norm, x_norm, x_norm, key_padding_mask=~mask.bool(), need_weights=False
        )
        y = x + self.attention_dropout(attent_emb)

        # PFF + skip-connection
        z = y + self.pff_dropout(self.pff(self.pff_norm(y)))

        return self.dropout(z)


class PositionwiseFeedForwardFast(torch.nn.Module):
    """
    Implements FFN equation with different activation functions.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        acceleration_config: dict = None,
    ) -> None:
        """
        :param d_mode: Embedding dimension.
        :param d_ff: Feed forward dimension, usually 4*d_model.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        :acceleration_config: Parameters for acceleration.
        """
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

        if acceleration_config["act_fn"] == "gelu":
            self.activation = torch.nn.GELU()
        elif acceleration_config["act_fn"] == "silu":
            self.activation = torch.nn.SiLU()
        elif acceleration_config["act_fn"] == "gelu_pytorch_tanh":
            self.activation = torch.nn.GELU(approximate="tanh")
        elif acceleration_config["act_fn"] == "relu":
            self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor.

        :returns: Position wised output.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForward(torch.nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        :param d_mode: Embedding dimension.
        :param d_ff: Feed forward dimension, usually 4*d_model.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        """
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor.

        :returns: Position wised output.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
