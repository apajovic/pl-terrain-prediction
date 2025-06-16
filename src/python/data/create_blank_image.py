from PIL import Image
new_im = Image.new("L", (256, 256), (0))
new_im.save("data/base.png")