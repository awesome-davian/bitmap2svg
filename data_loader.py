import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from collections import Counter



class SVGDataset(data.Dataset):
    def __init__(self, root,  vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.      
        Args:
            root: image directory.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.img_root = self.root + 'bitmap/'
        self.cap_root = self.root + 'caption/'
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        img_list = os.listdir(self.img_root)
        cap_list = os.listdir(self.cap_root)
     
        vocab = self.vocab
        path = img_list[index]

        image = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        with open(os.path.join(self.cap_root, cap_list[index]), 'r') as f:
            caption = f.readline()

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(os.listdir(self.img_root))


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_loader(root, vocab, transform, batch_size, shuffle, num_workers):
    svg = SVGDataset(root=root,
                       vocab=vocab,
                       transform=transform)
    
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=svg, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(root , threshold=0):
    """Build a simple vocabulary wrapper."""
    
    cap_root = root + 'caption/'
    cap_list = os.listdir(cap_root)
    counter = Counter()
    for i, id in enumerate(cap_list):
        with open(os.path.join(cap_root, id), 'r') as f:
            caption = f.readline()
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

   
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

