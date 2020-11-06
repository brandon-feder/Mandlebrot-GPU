# Converts the r, b, b integers passed to an integer
def rgbToInt(r, g, b):
    rgb = r
    rgb = (rgb << 8) + g
    rgb = (rgb << 8) + b
    return rgb
