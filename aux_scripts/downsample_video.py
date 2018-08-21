import cv2

video_path = '/hdd/UCF-ARG/rooftop_clips_stabilized/running/person07_02_rooftop_running.avi'
video_path_out = '/hdd/UCF-ARG/person07_02_rooftop_running_downsampled.avi'

capture = cv2.VideoCapture(video_path)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

fps = capture.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps // 2, (frame_width, frame_height))


# extract features
while capture.isOpened():
    flag, frame = capture.read()
    flag, frame2 = capture.read()

    if flag:
        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop
    else:

        break

        # When everything done, release the video capture and video write objects
    capture.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
