from PIL import Image
def handle_image(img):
    pass
    #print(im.tell())
    if (img.tell() == 1):
        img.save("frame" + str(img.tell()) + ".png")

im = Image.open("imsrc/cat_bubbles.gif")
try:
    while 1:
        handle_image(im)
        im.seek(im.tell() + 1)
except EOFError:
    pass # end of sequence
