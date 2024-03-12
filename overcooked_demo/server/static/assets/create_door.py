from PIL import Image, ImageDraw

# Create a new image for the door with white background
door = Image.new('RGB', (15, 15), 'white')
d = ImageDraw.Draw(door)

# Draw the door
d.rectangle([(0, 0), (14, 14)], fill=(139, 69, 19))
d.rectangle([(2, 2), (12, 12)], fill=(160, 82, 45))
d.ellipse([(11, 6), (12, 7)], fill=(128, 128, 128))

# Create a new image for the wall with white background
wall = Image.new('RGB', (15, 15), 'white')
d = ImageDraw.Draw(wall)

# Draw the wall in dark brown
d.rectangle([(0, 0), (14, 14)], fill=(102, 102, 102))

# Draw some bricks in a slightly darker brown
# d.rectangle([(0, 0), (14, 1)], fill=(81, 52, 28))
# d.rectangle([(0, 7), (14, 8)], fill=(81, 52, 28))
# d.rectangle([(0, 14), (14, 15)], fill=(81, 52, 28))
# d.rectangle([(7, 0), (8, 7)], fill=(81, 52, 28))
# d.rectangle([(7, 8), (8, 15)], fill=(81, 52, 28))

# Create a new image to concatenate door and wall with 1 pixel space between them
concatenated = Image.new('RGB', (31, 15), 'white')

# Paste the door and wall into the concatenated image
concatenated.paste(door, (0, 0))
concatenated.paste(wall, (16, 0))

# Save the concatenated image
concatenated.save('doors.png')
