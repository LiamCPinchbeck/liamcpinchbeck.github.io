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
We will look at common factors to be aware of when conducting these kinds of experiments in business and scientific settings before ever running an analysis or before tasking a data scientist to the task.
For those interested in the underlying workings of the tests, I also introduce standard practices and some more sophisticated techniques that you can employ going into as much detail while still keeping this post below a 60min reading time.

UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION
---


## Resources



- ["A/B Testing Course (Full Online Course)" -- Data36 - Online Data Science Courses](https://youtu.be/qhYsZWrTiuM?list=PLHS1p0ot3SVjQg0q1eEPrmOmPUY_AT1vB)
- ["Data Science Essentials – Crash Course in A/B Testing with Case Study" - freeCodeCamp.org](https://youtu.be/KZe0C0Qq4p0)
- ["A/B Testing in Data Science Interviews by a Google Data Scientist" - DataInterview](https://youtu.be/DUNk4GPZ9bw)

## Table of Contents

- [Statement of the problem](#statement-of-the-problem)
  - A button, landing page, medical tests, drug effectiveness
- [Structuring your hypothesis: Constructing a rigorous and sensible primary metric]
    - What is a primary metric
        - What do we REALLY want to measure?
    - Success metric: Clicks vs sales
    - Conversion rate
    - Metric Sensitivity
    - Key Questions you want to ask before going ahead with an A/B test
        - Big enough dataset
        - Long enough timeframe/large enough effect
        - Will it take too long --> Early Predictors
        - Hunches
        - One or more changes --> Buttons vs Vibes
        - A/B/C/D/E/F/G/H tests
    - What do you think the result will be? No tests, just hunches and predictions
- [Framing Results, and what can go wrong and how to avoid it]
    - [Peeking, multiple testing and observer bias]
    - [Objective Results, Subjective Interpretations]
        - Correlations/causation/confounding variables/indirect causation || "Looking backward"
            - Flickering
            - Returning customers and the novelty effect
        - Observer bias, cherry picking and p-value peaking
            - Finishing an A/B test experiment because the result you wanted is significant or discounting it because it doesn't support the "new shiny feature"
        - [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
    - A/A testing
    - 80% vs 99% --> 20x better
    - False positive, false negative (TABLE)
    - More _diagnostics_ the better, but don't get confused or cherry pick
    - Ethics
    - Power vs significance

- [A TLDR of going about an experiment]
- [The Mathematics of Null Hypothesis Testing]
    - [Basic two group tests]
    - [Two-way ANOVA]
    - [Viewing the experiment as a random walk (stochastic process)]
    - [CUPED control variates]
- [Bayesian A/B testing]
- [Final Rules of Thumb]




# Statement of the problem

It is quite common that when performing inference, a statistician doesn't know a priori how many parameters they need to describe her data or which one of a collection of models actually describes the data. This falls outside the typical region in which general sampling techniques (such as MCMC), apply and thus requires some specialised treatment in the form of 'trans-dimensional probability theory' and 'reversible-jump MCMC'.


# The Essence of the Problem that A/B testing is for


- Website button


# Footnotes
