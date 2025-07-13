import clip
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def text_to_image(text, model, ind, topk=4, clip_type="clip", tokenizer=None):
    with torch.no_grad():
        if (clip_type == "clip") or (clip_type == "long_clip"):
            text_tokens = clip.tokenize([text], truncate=True)
            text_features = model.encode_text(text_tokens.to(device))
        elif "bge" in clip_type:
            text_features = model.encode(text=text)
        else:
            prefix = "summarize:"
            text = prefix + text
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                max_length=80,
                truncation=True,
                padding="max_length",
            ).input_ids.to(device)

            text_features = model.encode_text(input_ids).to(torch.float)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype("float32")

        D, I = ind.search(text_embeddings, topk)
    return D, I