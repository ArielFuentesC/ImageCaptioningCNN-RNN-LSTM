# Image Captioning with Recurrent Neural Networks

## Overview

This repository consists of four distinct image captioning tasks, each employing a different model architecture. These tasks are as follows:

### Task 1: Simple Image Captioning

- A basic image captioning model utilizing a pre-trained CNN feature extractor and a single-layer RNN.
- Generates descriptive captions for images sourced from the COCO dataset.

### Task 2: Advanced Image Captioning

- A more sophisticated image captioning model featuring a two-layer RNN with GRU cell equations.
- Produces captions for COCO dataset images and is evaluated using the METEOR score.

### Task 3: LSTM-Powered Image Captioning

- An image captioning model equipped with a two-layer RNN using LSTM cell equations.
- Generates captions for COCO dataset images and evaluates performance using BLEU-4 and METEOR scores.

### Task 4: Attention-Enhanced Image Captioning

- The ultimate image captioning model, integrating a two-layer RNN, an attention mechanism, and a distinct CNN feature extractor.
- Captions COCO dataset images and evaluates performance with BLEU-4 and METEOR scores.

## Included Features

This repository includes scripts for various tasks, including model training, testing, and evaluation. Additionally, you'll find utility functions for data loading, preprocessing, and visualization.
