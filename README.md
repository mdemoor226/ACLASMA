# ACLASMA: Amplifying Cosine Losses for Anomalous Sound Monitoring in Automation

<figure>
  <img src="Extra/ACLASMA-Graphical-Abstract.png" alt="ACLASMA: Overview">
  <figcaption> We use a slightly modified Wilkinghoff [13] backbone. The hybrid model transforms the audio into a single
    representative embedding vector then feeds it through a cosine loss during training. Our approach aims to boost such losses. 
    The Toy Examples (top right) illustrate the difference between training with and without a Cosine Shift. The Shift reduces the clutter and improves
    the example embedding. Intuitively, anomalies within a trained space will reside as isolated points relatively distant from
    clusters of normal sound samples. This can be used to formulate anomaly scores. </figcaption>
</figure>



If training is going right, the Warp Loss should slightly Bound the Default Loss from Above.
