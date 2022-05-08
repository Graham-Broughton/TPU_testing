#### TPU_testing

In this experiment I pitted TPU's and GPU's against eachother with varying batch sizes. I used the mnist and flowers databases.

I tried to maximise reusability in this code as I was rerunning batches very often so it should be easier for others to use.

#### Summary
In each test the TPU was faster, in the flowers database I needed to lower the batch size just to be able to compare the two. The most pronounced was in the flowers db used pretrained Xception.

Thanks for stopping by!