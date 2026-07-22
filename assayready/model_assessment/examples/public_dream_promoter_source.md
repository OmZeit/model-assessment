# Public DREAM Promoter Example

Source dataset: HuggingFaceBio/random-promoter-dream-2022, config `supervised`.
Original source: Random Promoter DREAM Challenge 2022, Zenodo DOI `10.5281/zenodo.10633252`.
Direct files used when available: Zenodo record `10633252` `train.txt` and `val.txt`.
License noted by the Hugging Face dataset card: CC BY 4.0.

This script uses public measured promoter activity rows and trains a local random-forest baseline
when scikit-learn is available, otherwise a bootstrapped ridge baseline,
only to create model_prediction and model_uncertainty columns for exercising AssayReady.
Do not market these predictions as a best-in-class model.
