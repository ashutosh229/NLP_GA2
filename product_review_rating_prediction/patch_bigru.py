import json

path = r'c:\Users\kishor\Pictures\nlp\nlpga2\NLP_GA2\product_review_rating_prediction\kaggle_main.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update encode_tokens so empty text gets at least one <UNK>
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'def encode_tokens(' in src:
            new_lines = []
            for line in cell['source']:
                if 'ids = [word2idx.get(t, unk) for t in toks[:max_len]]' in line:
                    new_lines.append(line)
                    new_lines.append("    if len(ids) == 0: ids = [unk]  # prevent all-PAD row\n")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
            print('Updated encode_tokens cell')

# 2. Update AdditiveAttention
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'class AdditiveAttention' in src:
            new_lines = []
            for line in cell['source']:
                if "scores.masked_fill(pad_mask, float('-inf'))" in line:
                    new_lines.append("            all_pad = pad_mask.all(dim=-1, keepdim=True)\n")
                    new_lines.append("            safe_mask = pad_mask & (~all_pad)\n")
                    new_lines.append("            scores = scores.masked_fill(safe_mask, -1e4)\n")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
            print('Updated AdditiveAttention cell')

# 3. Ensure embedding_matrix has no NaNs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'np.save(' in src and 'embedding_matrix' in src:
            new_lines = []
            for line in cell['source']:
                if 'print(f\'Embedding matrix shape:' in line:
                    new_lines.append("embedding_matrix = np.nan_to_num(embedding_matrix, copy=False)\n")
                    new_lines.append(line)
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
            print('Updated embedding_matrix cell')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('kaggle_main.ipynb updated successfully.')

