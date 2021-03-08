import torch
import torch.nn.functional as F
from nltk.tokenize import WordPunctTokenizer
from torchtext.data import Dataset

from Model import Transformer, EncoderLayer, DecoderLayer, Node

tokenizer_W = WordPunctTokenizer()
def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def find_path(tree):
    path = []
    for nodes in reversed(tree):
        if len(path) == 0:
            path.append(nodes[0])
        else:
            parent_id = path[-1].parent_id
            for node in nodes:
                if node.id == parent_id:
                    path.append(node)
    return path

def find_best_path(tree):
    best = []
    for nodes in reversed(tree):
        if len(best) == 0:
            best.append(nodes[0])
        else:
            nodes_eos = []
            parent_id = best[-1].parent_id
            for node in nodes:
                if node.eos:
                    nodes_eos.append(node)
                if node.id == parent_id:
                    best.append(node)
            if len(nodes_eos) > 0:
                candidates = sorted([best[-1], *nodes_eos],
                                    key=lambda node: node.logps,
                                    reverse=True)
                candidate = candidates[0]
                if candidate.eos:
                    best = [candidate]
    return best


class Translator:
    def __init__(self, model_path, en_field_path, ru_field_path):
        self.D_MODEL = 256
        self.N_LAYERS = 2
        self.N_HEADS = 8
        self.HIDDEN_SIZE = 512
        self.MAX_LEN = 50
        self.DROPOUT = 0.25
        self.BATCH_SIZE = 64
        self.GRAD_CLIP = 1.0
        self.BEAM_SIZE = 1
        
        self.model_path = model_path
        
        self.EN_TEXT = torch.load(en_field_path)
        self.RU_TEXT = torch.load(ru_field_path)
        self.device = torch.device('cpu')

        self.model = Transformer(
            encoder=EncoderLayer(
                vocab_size=len(self.EN_TEXT.vocab),
                max_len=self.MAX_LEN,
                d_model=self.D_MODEL,
                n_heads=self.N_HEADS,
                hidden_size=self.HIDDEN_SIZE,
                dropout=self.DROPOUT,
                n_layers=self.N_LAYERS
            ),
            decoder=DecoderLayer(
                vocab_size=len(self.RU_TEXT.vocab),
                max_len=self.MAX_LEN,
                d_model=self.D_MODEL,
                n_heads=self.N_HEADS,
                hidden_size=self.HIDDEN_SIZE,
                dropout=self.DROPOUT,
                n_layers=self.N_LAYERS
            ),
            src_pad_index=self.EN_TEXT.vocab.stoi[self.EN_TEXT.pad_token],
            dest_pad_index=self.RU_TEXT.vocab.stoi[self.RU_TEXT.pad_token]
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        
    
    def translate(self, sentences):
            src_field = self.EN_TEXT
            dest_field = self.RU_TEXT
            beam_size = self.BEAM_SIZE
            max_len = self.MAX_LEN
            device = self.device

            if isinstance(sentences, list):
                sentences = [*map(src_field.preprocess, sentences)]
                targets = None
            if isinstance(sentences, Dataset):
                targets = [*map(lambda example: ' '.join(example.trg), sentences.examples)]
                sentences = [*map(lambda example: example.src, sentences.examples)]
            
            data = [*map(lambda word_list: src_field.process([word_list]), sentences)]

            translated_sentences, attention_weights, pred_logps = [], [], []
            self.model.eval()
            with torch.no_grad():
                # for i, src_sequence in tqdm.tqdm(enumerate(data), total=len(data), position=0, leave=True):
                for i, src_sequence in enumerate(data):
                    src_sequence = src_sequence.to(device)
                    src_mask = self.model.make_src_mask(src_sequence)
                    src_encoded = self.model.encoder(src_sequences=src_sequence, src_mask=src_mask)
                    tree = [[Node(token=torch.LongTensor([dest_field.vocab.stoi[dest_field.init_token]]).to(device), states=())]]
                    for _ in range(max_len):
                        next_nodes = []
                        for node in tree[-1]:
                            if node.eos: # Skip eos token
                                continue
                            # Get tokens that're already translated
                            already_translated = torch.LongTensor([*map(lambda node: node.token, find_path(tree))][::-1]).unsqueeze(0).to(device)
                            dest_mask = self.model.make_dest_mask(already_translated)
                            logit, attn_weights = self.model.decoder(dest_sequences=already_translated, src_encoded=src_encoded,
                                                    dest_mask=dest_mask, src_mask=src_mask) # [1, dest_seq_len, vocab_size]                      
                            logp = F.log_softmax(logit[:, -1, :], dim=1).squeeze(dim=0) # [vocab_size] Get scores                    
                            topk_logps, topk_tokens = torch.topk(logp, beam_size) # Get top k tokens & logps                    
                            for k in range(beam_size):
                                next_nodes.append(Node(token=topk_tokens[k, None], states=(attn_weights,),
                                                    logp=topk_logps[k, None].cpu().item(), parent=node,
                                                    eos=topk_tokens[k].cpu().item() == dest_field.vocab[dest_field.eos_token]))
                        if len(next_nodes) == 0:
                            break
                        next_nodes = sorted(next_nodes, key=lambda node: node.logps, reverse=True)
                        tree.append(next_nodes[:beam_size])
                    best_path = find_best_path(tree)[::-1]
                    # Get the translation
                    pred_translated = [*map(lambda node: dest_field.vocab.itos[node.token], best_path)]
                    pred_translated = [*filter(lambda word: word not in [
                        dest_field.init_token, dest_field.eos_token
                    ], pred_translated)]
                    translated_sentences.append(' '.join(pred_translated))
                    # Get probabilities
                    pred_logps.append(sum([*map(lambda node: node.logps, best_path)]))
                    # Get attention weights
                    attention_weights.append(best_path[-1].states[0].cpu().numpy())
                sentences = [*map(lambda sentence: ' '.join(sentence), sentences)]
            return translated_sentences


if __name__ == "__main__":
    model_path = 'transformer.pth'
    en_field_path = 'EN_TEXT.field'
    ru_field_path = 'RU_TEXT.field'
    translator = Translator(model_path, en_field_path, ru_field_path)
    translation = translator.translate(['it is fitted with a living room , sound - proof walls , flat - screen tvs , minibar and a safety deposit box .'])
    print(translation)