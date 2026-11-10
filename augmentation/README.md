# Style-transfer augmentation (experimental)

Sentinel's detector is trained on clear footage. The idea behind this module was to use neural style transfer to turn clean frames into harder ones, with clouds, rain and occlusion, so the detector sees more bad-weather conditions.

**Status: an experiment, not part of the pipeline.** It works as a standalone style-transfer tool, but it was never used to generate Sentinel training data because it isn't optimised enough: each image is produced by running gradient descent on the image itself, which is slow, especially on a CPU. Nothing here has been measured against the detector, so no accuracy gain is claimed.

## How it works

An ImageNet-pretrained VGG-19 is used as a fixed feature extractor:

- **Content loss:** mean-squared error at `conv4_2`, which keeps the scene layout (and so the bounding boxes) roughly in place.
- **Style loss:** mean-squared error between normalised Gram matrices at `conv1_1` through `conv5_1`, which carries over the texture and colour of the style image.
- **Optimisation:** gradient descent updates the generated image; the VGG-19 weights stay frozen.

For augmentation, the "style" image would be a photo of rain, cloud or fog, and the "content" image a Sentinel training frame.

## Run it

```bash
cd augmentation
python -m pip install -e .[dev]
style-transfer --content frame.jpg --style rain.jpg --output frame_rain.png --steps 200 --image-size 256
style-transfer-ui          # local Gradio interface
python -m pytest           # small math and config checks
```

The first run downloads the VGG-19 weights (about 550 MB). Use `--help` for content/style balance, learning rate and device.

## Layout

```text
augmentation/
  src/neural_style_transfer/
    core.py       VGG feature extraction, Gram matrices, optimisation
    cli.py        command-line interface and image I/O
    frontend.py   local Gradio browser interface
  tests/          small mathematical and configuration checks
```

Moved here from the former standalone `style-transfer-cnn` repository.

## Next steps if this is picked up again

- Replace per-image optimisation with a feed-forward or fast style-transfer network.
- Keep labels valid: check that bounding boxes still line up with the stylised image.
- Measure detector accuracy on weather-augmented validation data before using it for training.
