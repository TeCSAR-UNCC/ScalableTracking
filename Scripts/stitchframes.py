def save():
    os.system("ffmpeg -r 1 -i %01d.png -vcodec mpeg4 -y movie.mp4")