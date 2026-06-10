from tokenizers import Tokenizer
from tokenizers.normalizers import NFKC
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

def tokenizer_training_generator(ds, n_examples):
    for i in range(min(n_examples, len(ds))):
        story = ds[i]["text"].strip()
        yield f"<|user|>\nWrite a short story.\n<|assistant|>\n{story}\n<|end|>\n"

def tokenizer_trainer(train_ds, vocab_size:int, tokenizer_train_examples:int, special_tokens:list, ):
    normalizer = NFKC()
    pre_tokenizer = ByteLevel(add_prefix_space=False) # to prevent the first word to have a space like ĠHello

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )  
    # starts training the tokenizer
    tokenizer.train_from_iterator(
        tokenizer_training_generator(train_ds, tokenizer_train_examples),
        trainer=trainer,
        length=min(tokenizer_train_examples, len(train_ds))
    )

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    # adds a postproecessing step
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer




