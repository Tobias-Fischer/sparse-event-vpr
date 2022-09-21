# How Many Events Do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels.

This repository contains code for our paper "How Many Events Do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels". Currently it is a placeholder and contains some additional results that did not fit into the paper. We will provide the code upon paper acceptance.

If you use this code, please refer to our [paper](https://arxiv.org/abs/2206.13673):
```bibtex
@article{Fischer2022Sparse,
    title={How Many Events do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels},
    author={Tobias Fischer and Michael Milford},
    journal={arXiv 2206.13673},
    year={2022},
}
```


## Additional results

### QCR-Event-VPR results
The general trend of experiments on the QCR-Event-VPR dataset follows those observed on the Brisbane-Event-VPR dataset: Our method performs competitively while being significantly faster at inference time. Specifically, below we provide the recision-recall curves for one traverse pair where the robot travelled at the same speed, another with a moderate speed variation, and one with a large variation in speeds between the query and reference traverses:

![pr_curves_150_pixels_saliency_trial1_seqlen5_with_legend](https://user-images.githubusercontent.com/5497832/191398489-12213bc8-d6a9-44a9-8993-2686be273887.svg)


![pr_curves_150_pixels_saliency_trial1_events_per_frame_450_seqlen5_with_legend](https://user-images.githubusercontent.com/5497832/191399245-d2a132f7-15f0-495d-93d2-c3d7668c3c6e.svg)

![pr_curves_150_pixels_saliency_trial1_events_per_frame_450_seqlen5_with_legend](https://user-images.githubusercontent.com/5497832/191399361-22190e28-3ad1-49be-ac1f-7accd8806354.svg)


### Comparison between event data and native DAVIS RGB frames
Here wr briefly compare to conventional camera based place recognition. We note that in our previous paper [(Event-Based Visual Place Recognition With Ensembles of Temporal Windows)](http://doi.org/10.1109/LRA.2020.3025505), we conducted a study where we compared the performance of the proposed method in [6], i.e. reconstructing conventional images from the event stream and applying NetVLAD in these reconstructions, with NetVLAD directly applied on the native RGB frames that are provided by the DAVIS346 camera (the DAVIS346 can simultaneously record events and RGB frames). Noting that the native RGB frames provided by the DAVIS346 are of relatively low quality, in the previous paper we have shown that reconstructed frames from the event streams (black bar) outperformed the native RGB frames (blue bar). The relevant figure is provided here for reference:

Out of scientific curiosity, we conducted a similar experiment for our sparse pixel selection. Specifically, we have applied the methodology of probabilistically selecting sparse pixels to the native RGB frames. Instead of using the variance in the event counts, we simply used the variance in the intensity of the native RGB frames in the selection process. As in our previous paper, we found that using the native RGB frames of the DAVIS camera (green line in the figure below) led to lower performance compared to using the event-based approach (orange line in the figure below). We also found that the sparse pixels can be used as place recognition descriptors, even in the case of RGB frames (red line in the figure below). Indeed, just like in the event-based case (purple line), the pixel subsampling led to similar performance as using all pixels, while being computationally more efficient.
The relevant precision-recall curve is the following:

![pr_curves_150_pixels_saliency_trial4_useframes_seqlen5_with_legend](https://user-images.githubusercontent.com/5497832/191399083-10d68ff4-b16a-409e-9452-b899cf3cecbe.svg)
