from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModel


def make_bert_emb(
        df: pd.DataFrame,
        bert_name: str,
        out_path: Path,
        tokenize_fn: Callable[[PreTrainedTokenizer, pd.Series], torch.Tensor],
        emb_type: str = 'avg_max',
        n_shuffle: Optional[int] = None  # It's for tags...
):
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name).eval().cuda()

    save_as_npy = out_path.is_dir()

    results = pd.DataFrame()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        token = tokenize_fn(tokenizer, row).unsqueeze(0).cuda()
        with torch.no_grad():

            if n_shuffle is not None:
                token = token.squeeze()
                cls_token = token[0].reshape(-1)
                sep_token = token[-1].reshape(-1)
                token = torch.stack([
                    torch.cat((
                        cls_token,
                        token[1:-1][torch.randperm(len(token) - 2)],
                        sep_token,
                    ))
                    for _ in range(n_shuffle)
                ], dim=0)

            outputs = model(token)
            if emb_type == 'cls_token':
                emb = outputs[0][:, 0].squeeze().cpu().numpy()
            elif emb_type == 'avg_max':
                emb = torch.cat((
                    outputs[0].mean(dim=1),
                    outputs[0].max(dim=1)[0],
                ), dim=-1).mean(0).cpu().numpy()
            else:
                raise Exception('unsupported emb_type')
        if save_as_npy:
            np.save(str(out_path / f'{idx}.npy'), emb)
        else:
            results = pd.concat((
                results,
                pd.DataFrame(
                    data=emb.reshape(1, -1),
                    index=[idx],
                ),
            ))
    if not save_as_npy:
        results.columns = [str(n) for n in range(results.shape[1])]
        results.to_parquet(str(out_path))


def tokenize(tokenizer: PreTrainedTokenizer, row: pd.Series, col: str):
    text = row[col] if row[col] is not None else ' '

    try:
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            # max_length=tokenizer.model_max_length,
            max_length=512,
        )
    except Exception as e:
        print(row.name)
        raise e

    return torch.tensor(tokens)
