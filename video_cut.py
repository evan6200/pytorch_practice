from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys

start_time=sys.argv[1]
end_time=sys.argv[2]
output_name=argv[3]
ffmpeg_extract_subclip("/home/evan/gopro_pic/2019.11.31/GOPR0102.MP4", start_time, end_time, targetname=output_name)
