# Image Captioning with Recurrent Neural Networks

The repository includes four tasks, each with a different image captioning model. The tasks are described below:

- Task 1: A simple image captioning model using a pre-trained CNN feature extractor and a single-layer RNN. The model generates captions for images in the COCO dataset.
- Task 2: A more complex image captioning model using a two-layer RNN and GRU cell equations. The model generates captions for images in the COCO dataset and is evaluated using the METEOR score.
- Task 3: Another image captioning model using a two-layer RNN and LSTM cell equations. The model generates captions for images in the COCO dataset and is evaluated using the BLEU-4 and METEOR scores.
- Task 4: A final image captioning model using a two-layer RNN, an attention model, and a different CNN feature extractor. The model generates captions for images in the COCO dataset and is evaluated using the BLEU-4 and METEOR scores.

Includes scripts for training, testing, and evaluating the models, as well as utility functions for data loading, preprocessing, and visualization.
