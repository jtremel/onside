# Onside

A computer vision web-app that tracks the ball in soccer videos using neural networks and object tracking.

## How It Works

Onside streams video data directly from YouTube and outputs a live visualization that tracks the ball. The system works by fusing together a few computer vision techniques into a single pipeline.

1) Object Detection: When the first frame of the video is streamed in, a pre-trained neural network is used to locate and tag all the objects in the image (e.g., players, objects on the field, the ball). This uses a single-shot detection algorithm to give enough speed for near real-time processing while maintaining accuracy in detection.

2) Object Tracking: Once the ball is located, the data are passed to a tracking algorithm, which builds models of both the appearance and motion of the ball and uses these models to follow it from frame to frame. Tracking is faster than detection since there is some prior information on where the ball is, what it looks like, etc. Additionally, tracking is resilient to occlusion (when the ball passes behind the player) because it preserves the ball's identity across frames, unlike detection.

These two stages interact dynamically to track as consistently as possible. If the camera shot changes or the tracker loses the ball, the system is forced to re-detect the ball.

## Dependencies

### Object detection and tracking

Detection is accomplished via a model from *TensorFlow*, optimized using Intel's *OpenVINO* library. Tracking is accomplished using *OpenCV*. Onside requires installation of both *OpenVINO* and *OpenCV*. *TensorFlow* does **not** need to be installed, since the *OpenVINO*-optimized models are included in the repository. All object inference runs through *OpenVINO*'s inference engine.

### Video Streaming

Streaming video from YouTube is accomplished using *pafy*. *Pafy* optionally depends on the *youtube-dl* package (Recommended).

### Deployment

Onside is built with *Flask*. *Flask* needs to be installed.

## Author

* **Josh Tremel** - [jtremel](https://github.com/jtremel)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
