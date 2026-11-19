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





# Statement of the problem

A/B testing is a widespread approach to tackling decision making in scienfitic and industry settings. If you are unfamiliar, on the surface all that it amounts to is quantitatively asking whether Option B (our _alternative hypothesis_) performs better than or is different to some option A. A typical example is something like you suspect that your website landing page isn't clear enough and lowering the fraction of visitors who buy the product. 
With a little help from AI, I've chocked up the two websites below.


{% include ab-landing-demo.html %}

The right is obviously better: it's clear what the site is for, what they're selling and more. It's so much better that it begs the question of why even do the A/B test? In this case likely we probably shouldn't bother. The benefit to using design B is so obviously better that we DO NOT need to run an A/B test. However, we'll use it as an example here nonetheless.


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






## Table of Contents

- [Statement of the problem](#statement-of-the-problem)
        
- [Part I: Running a good experiment](#part-i-running-a-good-experiment)
    - Intro
        - A/B testing (the design) vs null hypothesis testing (one way to analyse it)
        - Why randomise at all? Correlation, confounding and the counterfactual
    - [Designing the experiment before you collect data](#designing-the-experiment-before-you-collect-data)
        - The primary metric: what do we REALLY want to measure?
            - Success metric: clicks vs sales; conversion rate
            - Guardrail metrics: what must not get worse
            - Metric sensitivity
            - Ratio metrics (clicks per page view) vs per-user metrics
            - Heavy-tailed metrics: revenue and the one whale customer
        - Write down your hypothesis, prediction and decision rule first
        - Errors and power
            - False positive / false negative (TABLE)
            - Significance vs power; 80% vs 99% (a 20% vs 1% miss rate)
            - Minimum detectable effect
            - Worked example: "you need ~31k users per arm"
        - Practical questions before you start
            - Is the dataset big enough? Is the timeframe long enough?
            - Will it take too long? Early predictors / surrogate metrics
            - Randomisation unit: user vs session vs page view
            - Interference: when treatment leaks into control (social networks, marketplaces)
            - One change or many: buttons vs vibes
            - A/B/C/.../H tests
        - Ethics
    - Reading the result
        - What a p-value is (and the five things it isn't)
        - Effect sizes and confidence intervals over p-values
        - Statistical vs practical significance
        - Ship, don't ship, or iterate
    - What can go wrong
        - Peeking
        - Multiple testing
        - The winner's curse
        - When randomisation breaks
            - Sample ratio mismatch; A/A tests
            - Returning customers, novelty effects, flickering
        - Long-term effects: holdout groups and short-term metrics that mislead
        - Objective results, subjective interpretations
            - Segment analysis: "it worked on mobile users in Tasmania"
            - Simpson's paradox
            - Observer bias and cherry-picking
            - More diagnostics are better, but don't go fishing
    - Checklist: running an experiment

- [Part II: How it works and other fun asides](#part-ii-how-it-works-and-other-fun-asides)
    - The causal foundation
        - Potential outcomes (Rubin): the effect we can never observe
        - Why randomisation makes the difference in means unbiased
        - SUTVA, and cluster / switchback designs when it fails
    - Basic two-group tests
        - Which test when: proportions vs means vs counts
        - z-test, t-test, chi-squared test of independence, matched pairs
        - Fisher's exact test and the lady tasting tea
        - The Neyman–Pearson lemma: why likelihood ratios are optimal
        - Deriving the sample-size / power formula
        - Power by simulation: Monte Carlo when the formula doesn't apply
    - When the CLT lets you down
        - Heavy tails and slow convergence (simulation)
        - Permutation tests: exact inference from shuffling labels
        - Bootstrap confidence intervals
        - Mann–Whitney U, and what it actually tests
    - Ratio metrics and the delta method
        - Why naive standard errors are too small
        - Deriving the delta-method variance
    - Variance reduction
        - Stratification and post-stratification
        - CUPED as control variates
    - More than two groups: one-way ANOVA and the F-statistic (brief; links to the follow-up post)
    - Peeking, formally
        - The experiment as a random walk
        - How bad is it? Simulating false-positive rates under peeking
        - A physicist's view: the look-elsewhere effect and trials factors
        - Wald's sequential probability ratio test (a WWII story)
        - Fixes: group-sequential designs (O'Brien–Fleming)
        - Fixes: always-valid inference (mSPRT, confidence sequences)
    - Multiple testing corrections: Bonferroni vs Benjamini–Hochberg
    - The winner's curse, formally
        - Selection bias in estimated effects
        - Shrinkage: James–Stein and hierarchical models to the rescue
    - Bayesian A/B testing
        - Beta-binomial posteriors and P(B > A)
        - Expected loss as a stopping rule
        - Does Bayes fix peeking? (Not really)
        - Hierarchical priors from past experiments
    - Beyond fixed experiments: multi-armed bandits
        - Explore vs exploit, and regret
        - ε-greedy, UCB and Thompson sampling
        - When to use a bandit instead of an A/B test (and when not to)
    - Case study: from A/B tests to adaptive clinical trials
        - A simulated drug trial with early stopping
        - Response-adaptive randomisation (bandits in medicine)

- Final rules of thumb




# Part I: Running a good experiment


## Designing the experiment before you collect data







# Part II: How it works and other fun asides







## AI Statement

I've not used AI for any of the body text or code writing pertaining to actual A/B testing, but at times have used it to 
- chock up some quick website designs


# Footnotes
