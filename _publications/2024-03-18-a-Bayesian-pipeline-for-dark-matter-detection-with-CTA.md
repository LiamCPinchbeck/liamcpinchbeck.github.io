---
title: "GammaBayes: a Bayesian pipeline for dark matter detection with CTA"
collection: publications
category: manuscripts
permalink: /publication/2024-03-18-a-Bayesian-pipeline-for-dark-matter-detection-with-CTA
excerpt: 'This paper is about a Bayesian analysis package to analyse gamma ray event data to perform inference on dark matter model parameters.'
date: "2024-03-18"
venue: 'JCAP'
paperurl: 'https://arxiv.org/abs/2401.13876'
# citation: 'Liam Pinchbeck, Csaba Balazs, Eric Thrane (2024). &quot;GammaBayes: a Bayesian pipeline for dark matter detection with CTA.&quot; <i>JCAP</i>.'
---

We present _GammaBayes_, a Bayesian Python package for dark matter detection with the Cherenkov Telescope Array (CTA). GammaBayes takes as input the CTA measurements of gamma rays and a user-specified dark-matter particle model. It outputs the posterior distribution for parameters of the dark-matter model including the velocity-averaged cross section for dark-matter self interactions ⟨σv⟩ and the dark-matter mass mχ. It also outputs the Bayesian evidence, which can be used for model selection. We demonstrate GammaBayes using 525 hours of simulated data, corresponding to 108 observed gamma-ray events. The vast majority of this simulated data consists of noise, but 100000 events arise from the annihilation of scalar singlet dark matter with mχ=1 TeV. We recover the dark matter mass within a 95% credible interval of mχ∼0.96−1.07 TeV. Meanwhile, the velocity averaged cross section is constrained to ⟨σv⟩∼1.4−2.1×10<sup>−25</sup> cm<sup>3</sup> s<sup>−1</sup> (95% credibility). This is equivalent to measuring the number of dark-matter annihilation events to be NS∼1.1<sup>+0.2</sup><sub>−0.2</sub>×10<sup>5</sup>. The no-signal hypothesis ⟨σv⟩=0 is ruled out with about 5σ credibility. We discuss how GammaBayes can be extended to include more sophisticated signal and background models and the computational challenges that must be addressed to facilitate these upgrades. The source code is publicly available at [this https URL](https://github.com/LiamCPinchbeck/GammaBayes).
