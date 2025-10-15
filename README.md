# Simple Food Recognition Test.

We believe that Google’s Cloud Vision API can accurately identify common household ingredients sent as images with at least 70% accuracy, for both single-ingredient and multi-ingredient images

## Features

* Auto-selects top images for single and multi-ingredient foods.
* Batch review via a web interface to mark low-quality or incorrect images.
* Downloads curated images locally.
* Tests Google Cloud Vision API food recognition.
* Generates a detailed JSON report of results.

## Requirements

* Python 3.7+
* Google Cloud Vision API credentials (`GOOGLE_APPLICATION_CREDENTIALS` environment variable)
* API keys for Unsplash and Pexels (stored in `.env` file):

  ```
  UNSPLASH_ACCESS_KEY=your_unsplash_key
  PEXELS_API_KEY=your_pexels_key
  ```
* Required Python packages:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

Run the script:

```bash
python test.py
```

You'll be prompted to select a mode:

1. **Curate dataset**: Auto-select and review images.
2. **Run test**: Run tests on an existing dataset.
3. **Full workflow**: Curate dataset and run tests.

### Dataset

* Curated dataset is saved as `curated_dataset.json`.
* Downloaded images are stored in `test_dataset/`.

### Review Interface

* Opens a local browser page to review selected images.
* Click an image to mark it for removal.
* Click **Complete Review** to finalize selections.

## Output

* Test results are saved in `test_dataset/test_results.json`.
* Summary includes accuracy and pass/fail status for single and multi-ingredient images.

please note:
* Rate limiting is handled for Unsplash and Pexels.

