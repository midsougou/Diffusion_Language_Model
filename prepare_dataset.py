import torch
from torch.utils.data import Dataset

class TokenBlockDataset(Dataset):
      def __init__(self, ds, tokenizer, seq_length, stride):
          self.input_ids = []

          # formatting
          token_ids = []
          for text in ds["text"]:
              formatted = (
                  f"<|user|>\nWrite a short story.\n"
                  f"<|assistant|>\n{text.strip()}\n<|end|>\n"
              )

              token_ids.extend(tokenizer.encode(formatted))

          for i in range(0, len(token_ids) - seq_length, stride):
              input_chunk = token_ids[i:i + seq_length]
              self.input_ids.append(torch.tensor(input_chunk))

          del token_ids
      def __len__(self):
          return len(self.input_ids)

      def __getitem__(self, idx):
          return self.input_ids[idx]