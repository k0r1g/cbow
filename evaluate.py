#
#
#
import torch
import pickle


# --------------------------------------------------------------------------
# Removed global vocabulary loading. 
# The `topk` function will now use vocab from the passed model object.
# --------------------------------------------------------------------------
# vocab_to_int = pickle.load(open('./words_to_ids.pkl', 'rb'))
# int_to_vocab = pickle.load(open('./ids_to_words.pkl', 'rb'))


#
#
#
def topk(mFoo, word: str = "computer"):
  # Use the vocab_to_int mapping from the passed model object (mFoo)
  if word not in mFoo.vocab_to_int:
      print(f"\nWarning: Word '{word}' not in model vocabulary for topk evaluation. Skipping.")
      return
  idx = mFoo.vocab_to_int[word]
  
  # Check if index is within the bounds of the embedding matrix
  if idx >= mFoo.emb.num_embeddings:
       print(f"\nWarning: Word '{word}' index ({idx}) is out of bounds for embedding matrix size ({mFoo.emb.num_embeddings}). Skipping topk evaluation.")
       return
       
  vec = mFoo.emb.weight[idx].detach()
  with torch.no_grad():

    vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
    emb = torch.nn.functional.normalize(mFoo.emb.weight.detach(), p=2, dim=1)
    sim = torch.matmul(emb, vec.squeeze())
    top_val, top_idx = torch.topk(sim, 6) # Get top 6 (includes the word itself)
    
    print(f'\nTop 5 words similar to "{word}":')
    count = 0
    for i, current_idx_tensor in enumerate(top_idx):
      current_idx = current_idx_tensor.item()
      # Use the int_to_vocab mapping from the passed model object (mFoo)
      # Add check to ensure index is valid for the model's vocab list
      if current_idx < len(mFoo.int_to_vocab):
          current_word = mFoo.int_to_vocab[current_idx]
      else:
          current_word = f"<Index {current_idx} out of bounds>"
          
      # Skip printing the target word itself
      if current_word == word and count == 0: 
          continue
          
      current_sim = top_val[i].item()
      print(f'  {current_word}: {current_sim:.4f}')
      count += 1
      if count >= 5: break # Stop after printing 5 neighbours
