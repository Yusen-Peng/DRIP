from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image

"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP
python src/example_analysis/craft_runner.py
"""


model = Model.load("hezarai/CRAFT", device="cuda")
# image = load_image("/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/original_images/0c0a22bfd0da315a.jpg")
image = load_image("/users/PAS2912/yusenpeng/DRIP/src/example_analysis/stop_sign.png")


outputs = model.predict(image)
result_image = draw_boxes(image, outputs[0]["boxes"], bbox_color=(0, 0, 255))

# save the image
# result_image.save("/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/craft_boxes/0c0a22bfd0da315a.jpg")
result_image.save("/users/PAS2912/yusenpeng/DRIP/src/example_analysis/text_detected.png")
