from PIL import Image
def handle_image(img):
    #print(im.tell())
    #img.save("frame" + str(img.tell()) + ".png")

im = Image.open("cat_bubbles.gif")
try:
    while 1:
        handle_image(im)
        im.seek(im.tell() + 1)
except EOFError:
    pass # end of sequence
