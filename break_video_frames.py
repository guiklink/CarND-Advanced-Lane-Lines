import cv2
import sys


destination = sys.argv[1]
source = sys.argv[2]
time0 = int(sys.argv[3])

# print "Destination:", destination
# print "Source:", source
# print "Starting on:", time0

# vidcap = cv2.VideoCapture(source)
# vidcap.set(cv2.CAP_PROP_POS_MSEC,time0)      # just cue to 20 sec. position
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   cv2.imwrite(destination+"frame%d.jpg" % count, image)     # save frame as JPEG file
#   count += 1


vidcap = cv2.VideoCapture('project_video.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,24000)      # just cue to 20 sec. position
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("Frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1