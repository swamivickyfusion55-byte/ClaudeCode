# v11.2.143 PatchFocus: WORK IN PROGRESS, do not upload to the Space

Asked: a narrower pentagon focused on the dark patch, and the marked area cleaned throughout the video.

Done here (synthetic tests only): the dark patch is found relative to a fit of the surrounding skin, the pentagon is the smallest
one enclosing the main patch, a keep-out mask (value 64) for the floor, clothing and the navel, and a per-frame body floor clip in
the video.

Not shippable yet, open problems:
- the fill (LaMa) leaves a colour cast in the video: about 40 % of the patch still shows, against 10 % for v11.2.142;
- the tattoo clip regresses: 0.6 % -> 10.6 % of the marks left (the ring tattoo is not removed; the keep-out overlaps part 3);
- the per-frame floor clip is limited by the pose estimate's own error.
