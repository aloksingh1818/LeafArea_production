from PIL import Image, ImageDraw

# Create a blank white image
img = Image.new('RGB', (400, 300), color='white')
draw = ImageDraw.Draw(img)
# Draw a green ellipse (simulating a leaf)
draw.ellipse((100, 80, 300, 220), fill='green', outline='darkgreen')
# Draw a gray circle (simulating a calibration coin)
draw.ellipse((320, 240, 370, 290), fill='gray', outline='black')
img.save('sample_leaf.jpg')
print('sample_leaf.jpg created.')
