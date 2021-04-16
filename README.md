# Machine-Learning-And-Art
This repository is a collection of work created by Simon Schoenbeck for a UCARE 2020-2021 project.

##### Inputs
* Template movie: (video file preferable an .mp4)
* New Clips: (directory of clips preferably all .mp4)
##### Output
A new movie.mp4
##### Combining Styles
* [Replace Video] Keep template audio and replace video with new clips
* [Replace Clips] Keep template order and replace with new clips
* [Generate Movie with Full Clips] Train a model and generate novel timeline with new clips
* [Generate Movie with Edited Clips] Train a model and generate novel timeline with new clips
##### Full Process
* Convert all video files to mp4
* Process all files
* If [Replacing (Video or Clips)]
    * Iterate over the template data and find the best clips
    * Save edited clips to temp folder
    * Generate timeline
* Else If [Generate Movie]
    * Train model
    * If [Edited Clips] save edited clips to temp folder
    * Predict a new timeline
* Convert timeline into movie
    * Concat clips to movie
    * If [Replacing Video] write origional audio over movie


