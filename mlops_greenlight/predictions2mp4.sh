ffmpeg -y -framerate 25 -pattern_type glob -i "$1*.png" -c:v h264 -pix_fmt yuv420p -b:v 5M $2
