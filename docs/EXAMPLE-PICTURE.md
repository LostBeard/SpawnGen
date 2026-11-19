# The examples' pictures (example.png, example_inpaint.png)

`src/SpawnGen.Studio/wwwroot/examples/example.png` is the picture SpawnGen's example workflows load (Load Image `example.png`), as ComfyUI ships
its own `input/example.png`. The server copies it into its input folder at start when that folder has no `example.png` (an existing file is
kept); the browser engine adds it to the page's inputs.

It was made by SpawnGen itself on 2026-11-18 (NV1080, RX 590, SpawnGen server, SpawnDev.ILGPU.ML 5.8.22) with **FLUX.2 [klein] 4B**
(black-forest-labs/FLUX.2-klein-4B, **Apache-2.0**), node Flux2Klein: 512 x 512 (square, so the examples that crop to 512 x 512 keep the whole person), 4 steps, seed 5, 4-bit weights, prompt:

> A full-length studio photograph of a smiling young woman in a red wool coat, blue jeans and white sneakers, standing upright and facing the
> camera, arms relaxed at her sides, her whole body in frame from head to feet with space above and below, centered on a plain light grey seamless
> background, soft even lighting, sharp focus

The person is generated, not a real person. One picture serves every example that loads one: a whole person (pose and object detection), a
plain background (background removal, depth, image to 3D), and plain colours (upscale, image to image, inpainting, picture edit). The PNG keeps
the prompt it was made with in its text chunk.

## example_inpaint.png (the Inpainting example's mask)

`examples/example_inpaint.png` is example.png with the red coat made fully transparent (alpha 0, everything else alpha 255). Load Image's mask is
1 - alpha, so the Inpainting example (`inpaint.json`, prompt "a blue denim jacket") redraws the coat out of the box. Before 2026-11-19 that example
loaded example.png, which has no transparency: the mask was empty, the run "succeeded", and the coat came back unchanged.

Made 2026-11-19 from example.png with a colour key (red > 0.35 and more than 1.8 x green and blue), a 9-px max then 5-px min filter to close the
buttons and folds, rows above 84 cleared (the lips matched the key): 30,321 of 262,144 pixels (11.6%). Both engines seed it beside example.png
(BrowserEngine.ExamplePictures, ServerApp.CopyExamplePicture). The tests ExamplesTests.Every_example_picture_is_one_the_site_ships (an example that
uses Load Image's MASK output must name a picture whose mask is not empty) and The_inpainting_example_picture_has_a_mask guard it.
