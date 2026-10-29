## How to run

1. Open this notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU** (or another GPU).
2. Run cells from top to bottom. The first cells install packages and download the tokenizer and dataset.
3. The notebook defaults to all official splits (287,113 train / 13,368 validation / 11,490 test), 12 epochs, and may take several hours on a T4. It mounts Google Drive and saves epoch checkpoints there so training can resume after a runtime restart. Reduce split sizes only for a clearly labeled trial.
4. The final cell downloads a ZIP containing the best model, resumable optimizer/scaler checkpoint, tokenizer, full per-test predictions, metrics, report, and plots.

**Compute warning:** Full greedy decoding for 11,490 test examples and training on all 287,113 training articles can take a long time. 