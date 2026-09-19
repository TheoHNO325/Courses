
import json
from train_bpe import pretokenize
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re

class BPEtokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab,self.merges = vocab,merges
        self.special_tokens = special_tokens
        self.PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            self.spec_to_id = {}
            for i in range(len(self.vocab)):
                if self.vocab[i].decode("latin-1") in self.special_tokens:
                    self.spec_to_id[self.vocab[i]] = i
        self.anti_vocab = {}
        for i in range(len(self.vocab)):
            self.anti_vocab[self.vocab[i]] = i
        self.merges_dic = {}

        for i in range(len(self.merges)):
            self.merges_dic[self.merges[i]] = i
        
    @classmethod
    def from_files(cls, vocab_filepath,merges_filepath, special_tokens=None):

        with open(vocab_filepath, 'r') as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode('latin-1') for k, v in vocab_json.items()}

        with open( merges_filepath, 'r') as f:
            merges_json = json.load(f)
        merge = [(a.encode('latin-1'), b.encode('latin-1')) for a, b in merges_json]

        return cls(vocab,merge,special_tokens)
    
    def encode(self, text):
        encoded_text = []
        chunk = text
        if self.special_tokens:
            escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
            escaped_tokens.sort(key=len, reverse=True)
            split_pattern = "|".join(escaped_tokens)
            segments = re.split(f"({split_pattern})", chunk)
        else:
            segments = [chunk]
        # 这里的单个setment应该是一个预分词的最小边界，所以我们可以遍历每个segment去看它是否包含merge
        # 并且应该是encode过的，所以应该先encode？
        token_strings = []
        # print("segments", segments)

            # print(spec_to_id)

        for seg in segments:
            if not seg:
                continue
            # 如果 seg 是特殊 token（在 special_tokens 中），直接保留
            if self.special_tokens and seg in self.special_tokens:
                token_strings.append(seg)
            else:
                # 应用 PAT 匹配预分词 token
                for match in re.finditer(self.PAT, seg):
                    token_strings.append(match.group())
        # print("token str",token_strings)

        for segment in token_strings:
            # encoded_segment = b''
            # print("segment",segment)
            if self.special_tokens and segment in self.special_tokens:
                encoded_text.append(self.spec_to_id[segment.encode('utf-8')])
                # print("有special",segment)
            else:
                word = [bytes([b]) for b in segment.encode('utf-8')]
                # print("word", word)


                while True:
                    
                    pair_to_rank = {}
                    for i in range(len(word)-1):
                        b = (word[i],word[i+1])
                        rank = self.merges_dic.get(b)
                        if rank is not None:
                            pair_to_rank[b] = rank
                    new_word = []

                    if pair_to_rank:
                        pair_to_rank = sorted(pair_to_rank, key=pair_to_rank.get, reverse=False)
    
                        best_pair = pair_to_rank[0]
                        i = 0
                        while i < len(word):
                            if i < len(word)-1 and (word[i],word[i+1]) == best_pair:
                                new_word.append((word[i]+word[i+1]))
                                i += 2
                            else:
                                new_word.append(word[i])
                                i += 1
                        # print("new_word",new_word)
                    else: 
                        break

                    word = new_word
                    # cannot_merge = True
                    # new_word = []
                    # i = 0
                    # while i < len(word): 
                    #     if i < len(word)-1 and (word[i],word[i+1]) in self.merges:
                    #         new_word.append((word[i]+word[i+1]))
                    #         i += 2
                    #         cannot_merge = False
                    #     else:
                    #         new_word.append(word[i])
                    #         i += 1
                        # assert b"".join(new_word) == b"".join(word)
                # print("word", word)
                for j in range(len(word)):
                    encoded_text.append(self.anti_vocab[word[j]])
                    # for i in range(len(self.vocab)):
                    #     flag = False
                    #     if self.vocab[i] == word[j]:
                    #         encoded_text.append(i)
                    #         # print("找到了",i,word[j])
                    #         flag = True
                    #         break
                    # if flag == False:
                    #     print(f"{word[j]}没找到对应的vocab id")

        # print("我倒要看看encode成什么样了",encoded_text)
        return encoded_text
    
    def encode_iterable(self, iterable):
        encoded_text = []
        for line in iterable:
            encoded_text.extend(self.encode(line))
        return encoded_text

    def decode(self,ids):
        decoded_text = []

        for i in range(len(ids)):
            decoded_text.append(self.vocab[ids[i]])

        decoded_text = b''.join(decoded_text).decode("utf-8",errors='replace')
        return decoded_text
                        