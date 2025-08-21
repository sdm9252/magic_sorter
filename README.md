# Magic Sorter

Minimal demo for identifying trading cards in a webcam feed.

## Overview

The app captures frames from your webcam, detects trading cards with
[YOLO](https://github.com/ultralytics/ultralytics), tracks them across frames
using ByteTrack, rectifies each card to a top-down view and finally matches it
against a gallery of known cards using a combination of pHash and
[CLIP](https://github.com/openai/CLIP) embeddings stored in a FAISS index.

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Build a gallery directory containing reference images of cards you own
   (JPEG files). Run the demo:

   ```bash
   python -m magic_sorter.app /path/to/gallery
   ```

The application will open your webcam and overlay the best matching card name
above each detected card.
