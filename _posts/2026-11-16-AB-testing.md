---
title: "AB testing / Null Hypothesis testing: How do I know my experiment is producing an effect?"
date: 2126-04-21
permalink: /posts/2126/11/2126-11-AB-testing/
tags:
  - Introductory
header-includes:
  - \usepackage{amsmath}
---


Hi there, in this post we will go through 'AB testing' or 'null hypothesis testing', that essentially wants to determine whether some change or idea performs better or at least _differently_ compared to the original.
This includes seeing whether certain features on websites increase customer churn and whether a medicine has a significant effect during a drug trial.
We will look at common factors to be aware of when conducting these kinds of experiments in business and scientific settings before ever running an analysis or before putting a data scientist to the task.
For those interested in the underlying workings of the tests, I also go through the workings of standard techniques and some more modern techniques that you can employ going into as much detail without inflating this post beyond a 60min reading time.

UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION
---


## Resources


- ["A/B Testing Course (Full Online Course)" -- Data36 - Online Data Science Courses](https://youtu.be/qhYsZWrTiuM?list=PLHS1p0ot3SVjQg0q1eEPrmOmPUY_AT1vB)
- ["Data Science Essentials – Crash Course in A/B Testing with Case Study" - freeCodeCamp.org](https://youtu.be/KZe0C0Qq4p0)
- ["A/B Testing in Data Science Interviews by a Google Data Scientist" - DataInterview](https://youtu.be/DUNk4GPZ9bw)
- [Null hypothesis testing - StatQuest](https://youtu.be/0oc49DyA3hU)
- [p-values - StatQuest](https://youtu.be/vemZtEM63GY)
- [Statistical Power - StatQuest](https://youtu.be/Rsc5znwR5FA)
- [Power Analysis - StatQuest](https://www.youtube.com/watch?v=VX_M3tIyiYk)
- [p-hacking - StatQuest](https://youtu.be/HDCOUXE3HMM)
- [Hypothesis Testing in Statistics - Steve Brunton](https://www.youtube.com/watch?v=vVDahuv1bq8)
- [P-values: misunderstood and misused](https://arxiv.org/abs/1601.06805)
- [Fundamentals of p-values: Introduction](https://arxiv.org/abs/1603.03952v1)
- [Post-hoc α Hypothesis Testing and the Post-hoc p-value](https://arxiv.org/abs/2312.08040)
- [Null hypothesis significance tests: A mix-up of two different theories, the basis for widespread confusion and numerous misinterpretations](https://arxiv.org/abs/1402.1089)






## Table of Contents

- [Statement of the problem](#statement-of-the-problem)
        - A button, a landing page, medical tests, drug effectiveness
        - A/B testing (the design) vs null hypothesis testing (one way to analyse it)
- [Part I: Running a good experiment](#part-i-running-a-good-experiment)
    
    - Designing the experiment before you collect data
        - The primary metric: what do we REALLY want to measure?
            - Success metric: clicks vs sales; conversion rate
            - Guardrail metrics: what must not get worse
            - Metric sensitivity
        - Write down your hypothesis and prediction first
        - Errors and power
            - False positive / false negative (TABLE)
            - Significance vs power; 80% vs 99% (a 20% vs 1% miss rate)
            - Minimum detectable effect
        - Practical questions before you start
            - Is the dataset big enough? Is the timeframe long enough?
            - Will it take too long? Early predictors / surrogate metrics
            - Randomisation unit: user vs session vs page view
            - One change or many: buttons vs vibes
            - A/B/C/.../H tests
        - Ethics
    - What can go wrong
        - Peeking
        - Multiple testing
        - The winner's curse
        - When randomisation breaks
            - Sample ratio mismatch; A/A tests
            - Returning customers, novelty effects, flickering
            - Simpson's paradox
        - Objective results, subjective interpretations
            - Observer bias and cherry-picking
            - More diagnostics are better, but don't go fishing
    - Checklist: running an experiment

- [Part II: How it works and other fun asides](#part-ii-how-it-works-and-other-fun-asides)
    - Basic two-group tests
        - z-test, t-test, chi-squared test of independence, matched pairs
        - Deriving the sample-size / power formula
    - Variance reduction: CUPED as control variates
    - More than two groups: one-way ANOVA and the F-statistic (brief; links to the follow-up post)
    - Peeking, formally
        - The experiment as a random walk
        - A physicist's view: the look-elsewhere effect and trials factors
        - Fixes: group-sequential designs (O'Brien–Fleming)
        - Fixes: always-valid inference (mSPRT, confidence sequences)
    - Multiple testing corrections: Bonferroni vs Benjamini–Hochberg
    - Bayesian A/B testing
        - Beta-binomial posteriors and P(B > A)
        - Expected loss as a stopping rule
        - Does Bayes fix peeking? (Not really)
    - Case study: from A/B tests to adaptive clinical trials
        - A simulated drug trial with early stopping
    - Final rules of thumb





# Statement of the problem

A/B testing is a widespread approach to tackling decision making in scienfitic and industry settings. If you are unfamiliar, on the surface all that it amounts to is quantitatively asking whether Option B (our _alternative hypothesis_) performs better than or is different to some option A. A typical example is something like you suspect that your website landing page isn't clear enough and lowering the fraction of visitors who buy the product. 
With a little help from AI, I've chocked up the two websites below.


{% include ab-landing-demo.html %}

The right is obviously better, so much so that it begs the question of why even do the A/B test? In this case likely we probably shouldn't bother, the benefit to using design B is so obviously better that we DO NOT need to run an A/B test. However, we'll use it as an example here nonetheless.


Another common example is asking whether Drug B performs better than Drug A. 

{% include base_path %}

<!-- Option 1: your rawpixel image -->
<!-- <img src="https://images.rawpixel.com/image_400/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvam9iNjgxLTAxNjZhLXguanBn.jpg" alt="Cartoon pills" style="display:block; margin:1em auto; max-width:320px;"> -->

<!-- Option 2: two capsules -->
<!-- <img src="{{ base_path }}/files/BlogPostData/ab-testing/drug-ab-capsules.svg" alt="Drug A vs Drug B capsules" style="display:block; margin:1em auto; max-width:420px;"> -->

<!-- Option 3: randomised trial diagram -->
<img src="{{ base_path }}/files/BlogPostData/ab-testing/drug-ab-trial.svg" alt="Patients randomly split between Drug A and Drug B, then recovery compared" style="display:block; margin:1em auto; max-width:520px;">


Or whether a given car brand's self-driving car software is safer than another.


<iframe src="{{ base_path }}/files/BlogPostData/ab-testing/sticker-pile.html"
        title="Car company logos" loading="lazy" scrolling="no"
        style="width:100%; aspect-ratio:16/10; border:0; display:block; margin:1.5em 0;"></iframe>


All of these questions can be designed and analysed as an A/B test. All you need are two (or more) options you want to look at, a _measurable_ outcome, and a question of whether any difference we see between our options is real or just random.




# Part I: Running a good experiment










# Part II: How it works and other fun asides







## AI Statement

I've not used AI for any of the body text or code writing pertaining to actual A/B testing, but at times have used it to 
- chock up some quick website designs


# Footnotes
