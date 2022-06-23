#### TPUs vs. GPUs

I wrote an (article on medium)[https://medium.com/@broughton.graham/exploring-google-colabs-tpu-38f819d0cb6c] about TPU's, this is the code I used to perform my experiments.

I tried to maximise reusability in this code as I was rerunning batches very often so it should be easier for others to use.

#### Summary
In each test the TPU was faster, in the flowers database I needed to lower the batch size just to be able to compare the two. The most pronounced was in the flowers db used pretrained Xception.

Thanks for stopping by!
