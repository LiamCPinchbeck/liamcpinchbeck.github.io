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

As usual, if you don't like my explanations or want another perspective I've got some external resources that you can have a look at.


- ["A/B Testing Course (Full Online Course)" -- Data36 - Online Data Science Courses](https://youtu.be/qhYsZWrTiuM?list=PLHS1p0ot3SVjQg0q1eEPrmOmPUY_AT1vB) (video)
- ["Data Science Essentials – Crash Course in A/B Testing with Case Study" - freeCodeCamp.org](https://youtu.be/KZe0C0Qq4p0) (video)
- ["A/B Testing in Data Science Interviews by a Google Data Scientist" - DataInterview](https://youtu.be/DUNk4GPZ9bw) (video)
- [Null hypothesis testing - StatQuest](https://youtu.be/0oc49DyA3hU) (video)
- [p-values - StatQuest](https://youtu.be/vemZtEM63GY) (video)
- [Statistical Power - StatQuest](https://youtu.be/Rsc5znwR5FA) (video)
- [Power Analysis - StatQuest](https://www.youtube.com/watch?v=VX_M3tIyiYk) (video)
- [p-hacking - StatQuest](https://youtu.be/HDCOUXE3HMM) (video)
- [Hypothesis Testing in Statistics - Steve Brunton](https://www.youtube.com/watch?v=vVDahuv1bq8) (video)
- [P-values: misunderstood and misused](https://arxiv.org/abs/1601.06805) (arXiv article)
- [Fundamentals of p-values: Introduction](https://arxiv.org/abs/1603.03952v1) (arXiv article)
- [Post-hoc α Hypothesis Testing and the Post-hoc p-value](https://arxiv.org/abs/2312.08040) (arXiv article)
- [Null hypothesis significance tests: A mix-up of two different theories, the basis for widespread confusion and numerous misinterpretations](https://arxiv.org/abs/1402.1089) (arXiv article)
- [Statistical Inference (2nd Edition) - George Casella and Roger L. Berger](https://pages.stat.wisc.edu/~shao/stat610/Casella_Berger_Statistical_Inference.pdf) (textbook)
- [What is the Peeking Problem?](https://confidence.spotify.com/glossary/peeking-problem) (blog post)



# Statement of the problem

A/B testing is a widespread approach to tackling decision making in scientific and industry settings. If you are unfamiliar, on the surface all that it amounts to is quantitatively asking whether Option B performs better than or is different to some option A. A typical example is something like you suspect that your website landing page isn't clear enough and lowering the fraction of visitors who buy the product. 
With a little help from AI, I've chocked up the two websites below.


{% include ab-landing-demo.html %}

The right is obviously better: it's clear what the site is for, what they're selling and more. It's so much better that it begs the question of why even do the A/B test? In this case we probably shouldn't bother. The benefit to using design B is so obviously better that we DO NOT need to run an A/B test. However, we'll use it as an example here nonetheless.


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
    - [Intro](#intro)
    - [Designing the experiment before you collect data](#designing-the-experiment-before-you-collect-data)
        - [Primary metrics: what do you ACTUALLY want to measure?](#primary-metrics-what-do-you-actually-want-to-measure)
        - [Writing down your hypothesis, prediction and decision rule](#write-it-down-first)
        - [Errors and power](#errors-and-power)
        - [Practical questions before you start](#practical-questions-before-you-start)
        - [Ethics](#ethics)
    - [Reading the result](#reading-the-results)
        - [What a p-value is (and the bajillion things it isn't)](#what-a-p-value-is-and-the-bajillion-things-it-isnt)
        - [Effect sizes and confidence intervals over p-values](#effect-sizes-and-confidence-intervals-over-p-values)
        - [Statistical vs practical significance](#statistical-vs-practical-significance)
        - [Ship, don't ship, or iterate](#ship-dont-ship-or-iterate)
    - [What can go wrong](#what-can-go-wrong)
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
    - [Checklist: running an experiment](#checklist-running-an-experiment)

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


## Intro

Suppose we've had version A of the landing page up for a month, then switch to version B for a month. Version A converted 3.6% of visitors, version B converted 4.1%. Does this mean version B wins?

Not necessarily, broadly for two reasons:

1. **Noise.** Even if the two pages were identical, two months of visitors would never convert at exactly the same rate. We need a way of asking whether the 0.5 percentage points is large enough that it's unlikely to be random chance.
2. **Confounding.** The months differ in more than just the page. Maybe B's month included Christmas, a sale, or a viral post that flooded the site with a different demographic compared to the month version A had. Any of those could produce the gap on their own.

These two problems are what an A/B test separates out:

- **The design** (randomly splitting visitors between A and B *at the same time*) mostly deals with confounding effects. Since a "coin flip" decides who sees which page, things like showing a given version over Christmas, sales and viral posts impact the results of the two versions in the same way, so on average the only systematic difference left should actually be the stuff we want to test.
- **The analysis** (e.g. a null hypothesis test) deals with noise: if the pages truly performed the same, how often would we see a gap at least this large?

These are two very different effects with their own limitations. "A/B testing" is often used to mean the statistical test, but the randomisation is getting us most of the way and no amount of clever analysis can fix a badly randomised experiment; the test is just one of several ways to analyse the data after performing the experiment (I'll try and cover Bayesian and sequential alternatives in Part II).





## Designing the experiment before you collect data



### Primary metrics: what do you ACTUALLY want to measure?

The most seemingly obvious thing that you have to do before running an A/B experiment to find which option is better is to actually ask what is "better"? 

For scientific purposes, “better” usually falls out of the question itself. “Does dark matter exist?” becomes “is the fraction of events attributed to a dark matter signal greater than zero, beyond what background fluctuations could produce?” “Is waveform model X better than Y?” becomes “does X fit the data significantly better than Y?”

For industrial settings "better" can be more subjective. We need to pick one or at most two "**primary metrics**" which are a single number that define "better" for a given experiment. Using website design as an example this could be:
- Higher click through rate from the landing page to the item page
- Higher click through rate from the item page to a specific item listing
- Higher number of people that visit a specific listing and then put the item into their cart
- Higher number of people that visit a specific listing and then actually buy the product
- Spending more time on the listing page
- Scroll more on the website

Some of these by their very nature are not good metrics. For example, "spending more time on the listing page" or "scroll more on the website" may not mean a "better" website, but rather a confused user that is trying to figure out what's going on! Click-through rates are less ambiguous but don't necessarily translate into sales. Sales is essentially what you _always_ want, however, the number of sales is likely to be significantly lower than the number of people just _visiting_ your website. Especially for smaller business, the rate at which people visit the website and buy an item will likely be so low that despite the metric being "good" we wouldn't be _sensitive_ to it. 

**Metric sensitivity** is whether given the amount of data we expect to collect in a given experiment and the expected number "hit rate" of the two options (fraction of people clicking through or fraction of people buying dog food), would we be able to get a significant enough difference to conclude that one option is better/different to the other? 

For example let's say over the course of a few months, you measure the fraction of people that once on the landing page and then actually buy a product on average 3%, but this number fluctuates by up to 0.4 percentage points (that is between 2.6% and 3.4%). Then, you want to perform an A/B test experiment for a single month for a new landing page, and you measure that the rate for the original page in this month is 2.8% and the rate for the new one is 3.1%. Do you think this would actually mean the new page is better? Or just that the number fluctuates enough that you can't really tell the difference based on the data from the experiment? We will later see that for common meaures of significance, that the required number of visitors would be around 50,000!

Another thing that we may want to keep in mind are what are called **guardrail metrics**. These are numbers that you don't want to *worsen* as part of your test. Switching to vaccine trials, your primary metric could be the infection rate (lower=better), and of course you want that number to go down. However, you **do not** want the number of people that experience adverse reactions to the vaccine to go up. Or in machine learning emotion classification tasks, where you use a machine learning model to predict the emotion showing in an image, a primary metric might be "correctly identifies the emotion on the face". *However*, you likely wouldn't want the rate at confidently classifying the wrong emotion to rise as well (e.g. someone that is very angry is actually classified as calm and happy...). On the landing page, a guardrail might be the page load time, refund rate, or newsletter unsubscribes. A "flashier" page that sells slightly more, but doubles the return rate, ain't the better option.



Another thing that we need to keep in mind is that people can be weird. That is, sometimes we will encounter outliers that would otherwise skew our results. Suppose instead of purchase rate we use *revenue per visitor*. Most visitors spend \$0, most buyers spend \$40–60 on a bag of dog food… and then one breeder buys \$4,000 worth for their kennel. If they happen to land in group B, B "wins", and the result doesn't really reflect the actual page but that breeder.

Metrics like revenue are **heavy-tailed**: a handful of extreme values can dominate the average, so the mean can vary a lot and then needs more data to conclusively pin down. Some common fixes are:
- **Cap (winsorise)** extreme values, e.g. treat any order above the 99th percentile as if it were the 99th percentile.
- **Use a binary metric** alongside it ("did they buy at all?"), which a single outlier can't move much.
- **Use methods that don't lean on the mean**, like rank-based tests or the bootstrap (Part II).

Whatever you choose, we need to decide on it *before* looking at the data. Deciding to drop outliers after seeing which group they landed in is exactly what can lead to **cherry-picking**.[^cherrypicking]

[^cherrypicking]: The term "cherry-picking" literally refers to the practice of fruit harvesters that throw away the "bad cherries", only selling the good ones. Referring to how an analyst can look at their data and decide that some data is "bad" and throw it away. 


A subtler trap is the **unit** of the metric. Click-through rate can be computed per *page view* (total clicks ÷ total page views) or per *user* (did each user click at least once?). If we randomise users but compute the metric per page view, one visitor with 100 page views counts 100 times, and their page views aren't independent: they're all the same person making similar choices. Treating them as independent makes our uncertainty look smaller than it really is, so we'll "discover" significant differences more often than we should. The fix (the *delta method*) is in Part II; for now the rule of thumb is: **analyse at the same level you randomise**, or use a method that accounts for it.

So we have a single primary metric that unambiguously means "better", a few guardrails that we don't want to get worse, and a check that we'll actually have enough data to see a difference. With these, we can now write down exactly what we expect to happen.




### Write it down first

Before collecting any data, write down what you're testing, what you expect, and what you'll **do** with a given outcome. 
This is called **pre-registration**, and its point is that you can’t quietly move the goalposts once you’ve seen the results. Particle physics (and many other domains, but I was trained as a particle physicist) takes a related idea further with blind analysis: the signal region is hidden, or the result is shifted by an unknown offset, while every analysis choice is made. You only “unblind” once the method is frozen, so the result can’t be nudged towards what you want it to be.

| | Landing page | Vaccine efficacy | Drug dosage |
|---|---|---|---|
| **Question** | Does the clearer page sell more? | Does the vaccine reduce infection? | Does recovery improve as the dose increases? |
| **Null hypothesis** | Purchase rate is the same | Infection rate is the same as placebo | Dose has no effect: recovery is flat from placebo upward (slope β₁ = 0) |
| **Prediction** | B ≥ A by at least 0.3 pp | ≥ 50% fewer infections | Recovery rises with dose (β₁ > 0) |
| **Primary metric** | Purchase rate per visitor | Infection rate over 6 months | Recovery rate at 4 weeks, regressed on dose (0, 10, 20, 40 mg) |
| **Guardrails** | Refund rate, load time | Serious adverse events | Side-effect rate at each dose |
| **Sample size / duration** | Decided by a power calculation | Decided by a power calculation | Decided by a power calculation |
| **Decision rule** | Ship B if significant at 5% and guardrails hold | Proceed if efficacy is significant and safety acceptable | Reject β₁ = 0 at 5%, then pick the lowest dose with acceptable side effects |

> Note: Vaccine and dosage thresholds are illustrative; real trials and the numbers they use are dictated by regulators. Please do not actually use what I'm using as fact *😂*






### Errors and power
<!-- - False positive / false negative (TABLE)
- Significance vs power; 80% vs 99% (a 20% vs 1% miss rate)
- Minimum detectable effect
- Worked example: "you need ~31k users per arm" -->


Here we are going to run through the key numbers that anyone running one of these tests or experiments should be familiar with (and may have used a couple times already in the post): **false positive and negatives**, **significance**, **power**, and **minimum detectable effect**.

For an A/B experiment, sometimes the measured outcome isn't the same as the 'correct' outcome. 
Any test/experiment can reach the wrong conclusion, and there are two ways to do it. 
You may remember these from the COVID tests: a *false positive* says you're infected when you're not; a *false negative* says you're clear when you're not. 
An A/B test makes the same kinds of mistakes, but about the **whole experiment** rather than one person:

| | **B really is no different** (null true) | **B really is different** (null false) |
|---|---|---|
| **Test says "B is different"** | ❌ False positive (Type I error) | ✅ Correct detection |
| **Test says "no difference"** | ✅ Correct | ❌ False negative (Type II error) |

Note that individual participants "going the wrong way" (an unvaccinated person who never gets infected, a visitor to the clear page who doesn't buy) aren't errors in this sense. That's just the person-to-person variation, the *noise*, that the test has to see through. The errors are about the conclusion we draw from all of them together.

The two error rates get names:
- **Significance level (usually denoted $$\alpha$$)**: the false positive, or Type I error, rate we're willing to accept, typically 5%.
- **Power $$(1-\beta)$$** : the probability of catching a real effect of a given size, typically 80%; i.e. we accept missing a real effect 20% of the time. With **$$\beta$$** being the false negative or Type II error rate/probability

For the signficance level, another way or saying this is if B truly does nothing in comparison to A (i.e. they're equivalent), then at $$\alpha$$ = 5% roughly 1 in 20 experiments will still conclude that they are different. We could run 20 tests and expect one ‘discovery’. (later we'll talk about ways to combat this if it is an issue). 
In particle physics, a new particle is only claimed at "$$5\sigma$$", a p-value of about $$3\times10^{-7}$$. In other words, if there were no new particle, random noise fluctuations would only produce a result that extreme about once in ***3.5 million experiments***. 
Reaching that bar takes enormous amounts of data, which a typical business/single experiment neither has nor needs, so we'll mostly stick with 1 in 20 ($$\alpha=0.05$$), or 1 in 100 ($$\alpha=0.01$$) if we want to be *really* sure.


Power is then the rate at which we *would* see a difference if one really exists. 
With 80% power ($$(1-\beta)=0.8$$), if B truly beats A by that amount and we ran the experiment 10 times, on average 8 runs would detect it and 20% of the time, or twice, would wrongly conclude there's no difference ($$\beta = 0.2$$). Below is a gif that demonstrates these aspects as areas under distributions. 

The first row shows what we're actually measuring over a 1000 simulated experiments, where each experiment is a test that a company or scientist would do using $$N_\text{samples}$$. $$X$$ measures the number of conversions or 'success' for experiments [^effect].
The second row then shows the estimated converion rates (e.g. fraction of people that buy a product given that they were on the landing page for landing page design A or B) or simply the rate of success (fraction of $$N_\text{samples}$$ where an effect was seen) as well as pooled estimates wheree you presume the null hypothesis (that the conversion rates are the same).


[^effect]: The definition of success can change depending on perspective, it just means a given sample sees an effect.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/power_diagram_two_sided.gif" 
      style="width: 100%; height: auto; border-radius: 0px;">
</div>
<br>

The third row is then the differences between the estimated effects per experiment: if in a given experiment/test we measure the conversion for landing page A to be $$p_A=0.03$$ and the conversion rate for landing page B to be $$p_B=0.035$$, then the difference would be $$D=p_B-p_A = 0.005$$. And then the final row is just the difference distribution (row 3) but standardised. The ***significance is*** the fraction of area outside the critical values we've set for the experiment (in this case $$\alpha=0.05$$ translates into a standardised value of $$\sim\pm 1.96$$) and the ***power is** the fraction of area in the green distribution outside those bounds.

We could also do what's called a **on-sided test** where all you're interested in is whether the conversion rate is higher, e.g. you will only ship a feature if it performs better, and if it performs the same or worse then you don't care, you won't ship it either way. Thee the power and signifiance are the same, except we're only concerned with the areas *above* (the positions of which slightly change between the one-sided and two-sided tests).

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/power_diagram_one_sided.gif" 
      style="width: 100%; height: auto; border-radius: 0px;">
</div>
<br>



Another question we might ask ourselves is if the true effect is *smaller* than we planned for, our the power drops, then we need to decide on a **minimum detectable effect**.

The minimum detectable effect is relatively self-descriptive, it's the smallest effect that we'd decide is significant given our choices on power and significance (e.g. if the number of sales go up by X% amount then the change was for the better, or if the rate of infection went down by Y% then we can say that 'this vaccine' is better).

The catch is that noise only shrinks, and thus the required size of the effect decreases, slowly with more data. 
For a 3% conversion rate, the random wobble in the *difference* between two groups could start at ~0.76 and then:

| Visitors per version | Typical variation in the difference |
|---|---|
| 1,000 | ±0.76 pp |
| 10,000 | ±0.24 pp |
| 50,000 | ±0.11 pp |

It typically shrinks like $$1/\sqrt{n}$$. Doubling the amount of data you have decreases the measurement noise by a factor of 2, and increasing the amount of data by 10 decreases the amount of noise only by about a factor of 3...  So small effects need *a lot* of data. For example with the standard power and significance numbers used above and the "typical variation" table:

| MDE (from a 3% base rate) | Visitors needed per version |
|---|---|
| +0.3 pp (3.0% → 3.3%) | ~53,000 |
| +0.6 pp (3.0% → 3.6%) | ~14,000 |
| +0.9 pp (3.0% → 3.9%) | ~6,500 |

Halving the effect you want to detect roughly *quadruples* the sample you need (inverse of the decrease in the size of the noise). A plot showing how you probe more minute differences in effects (MDE) as a function of days presuming 2,000 visits per day, base conversion rate of 3%, $$alpha=0.05$$ (significance), and power of 80%. You can see that early on there is a large benefit to increasing the time period one or two days, but then later on the benfits significantly decrease.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/MDE_vs_days.png" 
      style="width: 80%; height: auto; border-radius: 0px;">
</div>
<br>

The standard formulae to decide how much data is required for a given MDE (denoted $$\delta$$ in the formula) are:

$$\begin{align}
N \approx \frac{(\sigma^2_\text{A} + \sigma^2_\text{B})(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta_\text{MDE}^2}
\end{align}$$

with $$\sigma^2_\text{A}$$ and $$\sigma^2_\text{B}$$ being the noise in the measurements squared or the *variance*. 

If you're specifically interested in conversion-rates for option A ($$p_A$$) and for option B ($$p_B$$) then we can use a more specific formula where $$\delta$$ is now the difference between the rates $$\delta_\text{MDE}=p_B - p_A$$ and the two variances become a function of $$p_A$$ and $$p_B$$:

$$\begin{align}
N \approx \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \,\big[p_A(1-p_A) + p_B(1-p_B)\big]}{(p_B - p_A)^2}
\end{align}$$

where $$z_{1-\alpha/2} \approx 1.96$$ for $$\alpha = 0.05$$ and $$z_{1-\beta} \approx 0.84$$ for 80% power. A demonstration of the kind of effects one can see if they don't properly increase the number of samples can be seen in the gif below.

<div style="text-align: center;">
  <img class="social" 
      src="/files/BlogPostData/ab-testing/n_samp_vs_mde_with_confusion.gif" 
      style="width: 80%; height: auto; border-radius: 0px;">
</div>
<br>

With three examples over three hundred simulated experiments: a quadratic increase in the number of samples (correct), a linear increase (e.g. one might assume half the change requires double the samples), or a constant amount (e.g. say someone says that they had more power than they needed in their last tests, surely the same amount will do for a little bit smaller of an MDE...nope). The left column is the number of samples used for experiments vs the MDE (the bottom two rows highlighted red because with our choice of power and significance, they are wrong).
The middle column shows the false positive and false negative rates.
And the last column shows the confusion tables for the experiments.

There are two things to note that can both be seen in the centre column. As your worsen the number of samples used: 1. the false positive rate doesn't go up and 2. the false negative rate *does* go up. 
Throwing away all the math really quickly, if we just have less information, then the relative noise to the changes will *increase*. 
Hence, you will be more likely to say that there is no significant difference most of the time, even when there is one (increase in false negative rate). But because you're less likely to say anything at all, you're also not going to say that there is a difference when there isn't one (no impact on false positives).

This made me wonder about what would lead to an increase in false positives but keep the false negative rate the same, and that basically falls under problems with peeking and multiple testing that I will go through in [What can go wrong](#what-can-go-wrong).


### Practical questions before you start


Before one even starts an AB test, I think that there a few questions one should ask particularly for business with relatively low traffic or budgets.

***Is the dataset big enough? Is the timeframe long enough? \|\|*** As shown in the percentage MDE given days figure (above the 'standard formula' for required number of samples for a given MDE) shown again below. Remembering that the option A conversion rate was $$\sim 3\%$$. 

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/MDE_vs_days.png" 
      style="width: 80%; height: auto; border-radius: 0px;">
</div>
<br>


We noted that increase the amount of days (/samples) early on can allow you to probe much smaller effect sizes / difference between version A and version B's. But later on, the difference in effect size becomes extremely small. 

Similarly, if your expected MDE is too small (e.g. in this case $$\sim 0.2\%$$ or $$0.002$$ difference in the conversion rate/fraction) then there's a chance you would have to expend a significant amount of time to robustly explore that effect. 


Generally, if the effect could only be seen after more than two months (the rough period in the figure) then we should ask ourselves whether it's worth expending the time and resources for such a small effect.  

And finally, there is the opposite case where even if you expect to be able to see a difference in a couple days, in the case of website changes/software features, we should generally still run the experiment for two weeks at minimum. This is to rule out periodic effects during the week. *For example*, the new features might make better landing pages, but they take an extra moment to load. On a weekend, maybe people wouldn't mind as much as they have more time to spend on the website and a moment here or there wouldn't matter. *However*, someone during the week might be pressed for time, and those extra few seconds might be frustrating enough to counteract the positive changes.



***Will it take too long? Early predictors / surrogate metrics \|\|*** In the case of the figure above, we are presuming that the conversions are from landing page to purchases (which 2,000 visitors per day is a large number already), but in most cases there would be significantly more people that go from the landing page to the list of items that you are selling for example. 

We could even argue that clicking to the item listing page is most of the battle and is also a reasonable predictor for how well the landing page is doing. In this case, despite the purchases being your primary predictor, it might be more advantageous to use an **early predictor** which could be people going from the landing page to the item listing page. In this case, you could explore smaller effects in a smaller timeframe, reducing cost and time spent. Below is a figure showing the size of the effect one is able to explore in this kind of case, presuming a base rate of 30% for going from the landing page to the item listing page, and 3% as before for the landing page to actual purchases.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/MDE_vs_days.png" 
      style="width: 80%; height: auto; border-radius: 0px;">
</div>
<br>

If we use an early predictor, we can explore differences of below 2% (same relative difference to our our original 3% to 0.2%) in roughly a seventh of the time! So TLDR: "is there a predictor that I could use that still indicates a difference in the effect I really want to explore, but that I could get more robust results quicker?"


***Small changes or big changes: buttons vs vibes \|\|*** A quick addition to the point on early predictors, for low traffic websites, it's almost always better to only A/B test large changes. You might have heard that Google A/B tests absolutely every feature of there websites down to the pixel size differences of the buttons, but they have the traffic to allow this. If you are reading this, you are likely not Google. For most websites, we will likely never be able to distinguish these kinds of effects, so it's better to put your budget towards larger tests (or 'vibes' of a website) or just other things to improve the business/effect that you're trying to look into.



***Randomisation unit: user vs session vs page view* \|\|** Another easy trap that these tests can fall into is presuming *independence* between samples. That is, that the information in one sample doesn't allow any extra information to be gained on another. An example of this might be a new software feature on a "smart boat". The company/manufacturer/software developers might be recording every time a button is pressed on a new dashboard based on age of the user/driver. You do your tests and find that it roughly decreases with age (but because of expected demographic of "smart" boat owners and tech usage), except for 75-80 year olds where there is a 1000% jump compared to 70-75 and 80-85 year olds. But it turns out this is because there is one person, I usually think of them as "Steve", just really loves pressing buttons and scuba diving, and spent 3 hours just pressing buttons on the new dashboard skewing your results. 


<div style="text-align: center;">
  <img 
      src="https://media1.tenor.com/m/6gXsVMHqFtQAAAAC/damn-you.gif" 
      style="width: 50%; height: auto; border-radius: 0px;">
</div>
<br>

However, if we had segmented each sample to whether a driver simply pressed a button or not, this wouldn't have happened. (And did I redo that whole example for the gif? Yes I did.)





***Interference: when treatment leaks into control (social networks, marketplaces) \|\|*** Similar to the above, you can have that during an experiment one group impacts the other. For example, in vaccine efficacy, you might be measuring infection rate for those that get a vaccine (group B) and those that don't (group A). There might be a real efficacy to the vaccine so the infection rate for group B drops, but because less people are infected in total, then those in group A are *also* less likely to get infected. Making the *actual* amount of efficacy harder to determine or under-estimated.

Or another example could be a discount for a new ride-share company/method (e.g. comparing standard Uber to UberCheap), but because more people are utilising the service, there are less total drivers, so there are less for group A. 

Or maybe you have a new advertising compaigns on social media, and one is doing so well that users picked to be part of the control group (i.e. seeing how often people engage with the company without seeing a campaign) are sent a link from a friend in the experiment (seeing the advertising) group. Biasing the results again. 


The general fix for these kind of problems is to randomise bigger groups, so treatment and control are less likely to interact. Using whole cities or regions (*cluster randomisation*), alternating time regions (*switchback experiments*, common to tests involving ride-share), or friend clusters. The problem with this is that there are fewer independent units/tests, so less power (but importantly *rigorous* power rather than a biased power), which links back to whether we have a large enough sample-size for the effect we're trying to investigate.




***A/B/C/D/E/F../../.. tests \|\| [Multiple Comparisons Problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)*** Another pit fall that we can run into is having a slew of ideas / features / concepts / drugs / vaccines that we wish to test, and we might think that instead of restricting ourselves to one or two at a time that we should test all of them and see what sticks. The problem with this approach is that you are then likely to 'discover' a positive result (seeing an effect/difference) even when there isn't, simply because of the number of tests you are running. 

We are in the land of probabilities, and if we set our significance level to $$\alpha=0.05$$, that means that if we run 20 experiments where *none* of them have a real, on average we would expect to see one "result". If we investigate 20 proposed vaccines, and even if none of them should actually lead to a difference, we would think that one of them is effective despite it not being one. For this level of signifiance and number of tests, the chance of at least one false positive is $$1- 0.95^{20} \approx 64\%$$! 


There are *multiple* ways to *try* and address this problem, each with it's own benefits and deficits. One of which is known as the [Sidak correction](https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction), that defines a new significance for the tests to control the family wise error (positive) rate. The new significance that applied for each comparison defined with respect to the original level of significance $$\alpha$$ is 

$$\begin{align}
\alpha_\text{new} = 1 - (1-\alpha)^{1/\text{num tests}}.
\end{align}$$

This is such that if you are running 20 tests that are independent (don't affect each other), then the false positive rate comes down to 

$$\begin{align}

(1-\alpha_\text{new})^{20} = (\alpha^{1/20})^{20} = \alpha,

\end{align}$$

i.e. the significance that we actually want. For each comparison though the required significance increase. For 20 tests, $$\alpha_\text{new} \approx 0.0026$$ per test ($$z_\text{crit} \approx 3.0$$ instead of $$\sim1.96$$), so for the same number of samples per test, you get less power.

This raises the key question of whether to test all these methods at once, or cam we use our intuition/expertise to narrow down our options such that we don't have to double the number of required samples (or more!) to get a robust result. We will go into some more multiple testing solutions and approachs in the [What Can Go Wrong](#what-can-go-wrong) section of the post.




### Ethics

Another thing that (almost) goes without saying is if there are any impacts on people's physical or mental health or any other possible ethical concerns, those should be addressed properly.





## Reading the results


Okay, so we've defined significance, false positive and negative rates and power and looked at some key questions we should ask before conducting a test such that we can set up or define a good experiment. Now let's pretend we have our results, how do we communicate them clearly, concisely and in a way that takes into account the likelihood of the result / doesn't over-promise.





### What a p-value is (and the bajillion things it isn't)



It is common to mix up the *significance level* we've defined for our test, and the more colloquial meaning of "*significance*": "If we assume our null hypothesis is true, how likely would it have been to observe the data we collected, or to observe even more extreme data". This quantity is called *p-value* and it is *p*lagued with problems.

First things first though, what is it and how does it compare to our previous values? Below should be a gif, with the top panel being the familiar descriptions of $$\alpha$$, $$\beta$$ and power with respect to the distribution of standardised differences or *z-scores*. The bottom panel shows the distribution of *p-values* for the case of the null hypothesis being true (uniform values) and the alternate hypothesis being true (values cluster around 0). We calculate the probability of the result if we assume that the null hypothesis is true, and thus how *unlikely* the results would be if the alternate hypothesis weren't true.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/ab-testing/power_diagram_two_sided.gif" 
      style="width: 100%; height: auto; border-radius: 0px;">
</div>
<br>

The significance of the result is then defined with respect to whether the p-value is smaller than our defined significance threshold $$\alpha$$. i.e. if $$p<\alpha$$ we can say that we can reject the null hypothesis / option A. Rejecting at $$\alpha=0.05$$ is the same as checking whether $$\|z\| > 1.96$$. That is all the p-value is: the probability of observing the data I got, or something more extreme, presuming that the null hypothesis/ option A=option B is true.


Things that the p-value is not:
- *not the probability the null is true*. This is a subtle one, but we measure probability of data given hypothesis, not hypothesis given data.
- *not the probability the result happened "by chance"*. This is the false positive rate $$\alpha$$, it is a quality of the testing process, not the data outputs of the process. The p-value is the probability of data, or something more extreme, given the null hypothesis. No alternate hypothesis is involved which would be required for this state.
- *says nothing about effect size or importance*. We'll tackle this one in the next section, but TLDR, a smaller p-value doesn't necessarily mean a larger observed effect.
- *p = 0.049 vs 0.051 isn't a meaningful difference*. The threshold that we set for ourselves, $$\alpha$$, is largely arbitrary and being either 0.1 percentage points below or above this threshold is not meaningful. With enough data, even a tiny, worthless effect gives a tiny p-value.
- *a non-significant result isn't evidence of no effect*. "Absence of evidence isn't evidence of absence". Just because you didn't get a significant p-value for an experiment/test, doesn't mean that the effect isn't there. Just that with the data we have, you can't tell.

TLDR: If your p-value lies below the significance threshold $$\alpha$$ you set *before* the experiment, then you can reject the null hypothesis.


### Effect sizes and confidence intervals over p-values

Another thing that you might need to report if you're modelling a new *relationship* or *size* of an effect, is what range of values the parameters or new effect size fall into. For this we utilise **[confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval)** which given some confidence threshold X%, then we can say that the probability of observing the data given that the parameters fall within this range is X%. 

More concretely, if you measuring the efficacy of a vaccine you could say 
>"With a probability of 68% we would observe the data we collected, or something more extreme, if the new effect size falls within 3% and 4%." 

This ([interval estimation](https://en.wikipedia.org/wiki/Interval_estimation)) can be more useful than simply reporting the average or a [*"point estimate"*](https://en.wikipedia.org/wiki/Point_estimation) of the results and clarifies that the given result has uncertainties.

Presuming the central limit theorem (that the mean estimates of our data approximately follow a Normal distribution), then a simple confidence interval for a reported mean value $$\bar{x}$$ for $$n$$ samples, with sample standard deviation of $$s$$ is given by:

$$\begin{align}
\text{Confidence Interval} = \left[\bar{x} - c \frac{s}{\sqrt{n}}, \bar{x} + c \frac{s}{\sqrt{n}} \right].
\end{align}$$

Where $$c$$ is the corresponding to the z-score (value of the parameter in standardised coordinates) for the X%. Some common ones are:
- $$c=1.645$$ for 90%
- $$c=1.96$$ for 95%
- $$c=2.44$$ for 98%
- and $$c=2.575$$ for 99%

You can read off more of these [here](https://www.craftonhills.edu/current-students/tutoring-center/mathematics-tutoring/distribution_tables_normal_studentt_chisquared.pdf) if you so wish. This also works for the $$z_{1-\alpha/2}$$ we were working with briefly above.



### Statistical vs practical significance






### Ship, don't ship, or iterate







## What can go wrong





### Peeking and Winner's Bias





### Multiple testing





### When randomisation breaks
    - Sample ratio mismatch; A/A tests
    - Returning customers, novelty effects, flickering





### Long-term effects: holdout groups and short-term metrics that mislead





### Objective results, subjective interpretations
    - Segment analysis: "it worked on mobile users in Tasmania"
    - Simpson's paradox
    - Observer bias and cherry-picking
    - More diagnostics are better, but don't go fishing


### The Replication Crisis

All of these problems have many real world consequences, not only on a per-experiment level, but for widespread research and development. Particularly, they combine into what is called the replication crisis


1. Low power plus filtering on p < 0.05 (the false discovery rate). Work this through yourself. Suppose 10% of the hypotheses a field tests are actually true, studies have 80% power, and α = 0.05. Out of 1,000 studies, how many true positives are there, how many false positives, and what fraction of the "significant" findings are real? Now drop power to 35%, which is typical of many small studies. What happens to that fraction? This is the core of Ioannidis's argument, and it uses nothing beyond α, power and your "not the probability the null is true" bullet.

2. The winner's curse (type M error). Underpowered studies that do reach significance overestimate the effect, as in your planned histogram. So even when the original finding is real, replications tend to find a smaller effect. The large psychology replication project found replication effects averaging about half the originals'. Your type M plot explains why in one picture.

3. Multiple comparisons and p-hacking. Trying several outcomes, subgroups or analysis choices, and reporting the one that worked, is your 64% calculation in disguise. Simmons, Nelson & Simonsohn showed that a few such flexible choices can push the false-positive rate from 5% to over 60%. This links to your pre-registration and blind-analysis sections, since those are the fixes.

4. Publication bias: journals publish p < 0.05 and file-drawer the rest. That's the filter that turns points 1–3 into a literature full of false and inflated findings.

5. The industry parallel, to keep it relevant to your readers: A/B test "wins" that vanish after launch are the same phenomenon. Kohavi and colleagues report that most ideas tested at large companies don't improve the metric, so the starting proportion of true effects is low, exactly the situation in point 1.




## Checklist: running an experiment






# Part II: How it works and other fun asides





## Test Statistic rules

$$X_A \sim \text{Binom}(n, p_A), X_B \sim \text{Binom}(n, p_B)$$

$$\hat{p}_A = X_A/n, \hat{p}_B = X_B/n$$

$$D = \hat{p}_B - \hat{p}_A$$

$$\mathbb{E}[\hat{p}] = p, \mathbb{V}[\hat{p}] = \frac{p(1-p)}{n}$$

$$\mathbb{E}[D] = p_B-p_A \equiv \delta, \mathbb{V}[D] = \frac{p_A(1-p_A)}{n} + \frac{p_B(1-p_B)}{n} = \sigma_1^2$$


By CLT: $$D \sim \mathcal{N}(\delta, \sigma_1^2)$$






---

Null Hypothesis ($$H_0$$): $$p=p_A=p_B$$

pooled estimates: $$\bar{p} = \frac{p_A+p_B}{2}$$

null-hypothesis standard dev: $$\sigma_0 = \sqrt{\frac{2\bar{p}(1-\bar{p})}{n}}$$

null-hypothesis z-score: $$z_0 = \frac{D}{\sigma_0}$$

rejection cut-off: $$c = z_c \sigma_0$$

Power: $$P(D>c) + P(D<-c)$$. From step 1 to step 2 we assume $\sigma_1$ is correct, then the left side of the equality follows a standard normal.

$$\begin{align}
P(D>c) &= P(\frac{D-\delta}{\sigma_1}>\frac{c-\delta}{\sigma_1})\\
&= 1- \Phi(\frac{c-\delta}{\sigma_1})\\
&= \Phi(\frac{\delta - c}{\sigma_1}) \\
&= \Phi(\frac{\delta - z_c \sigma_0}{\sigma_1})
\end{align}$$

Using $$\sigma_1=s_1/\sqrt{n}$$ and $$\sigma_0=s_0/\sqrt{n}$$

$$\begin{align}
z_{1-\beta} &= \frac{\delta - z_c s_1/\sqrt{n}}{s_0/\sqrt{n}} \\
z_{1-\beta} s_1 &= \delta \sqrt{n} - z_c s_0 \\
n &= \left(\frac{z_{1-\beta} s_1 + z_c s_0}{\delta} \right)^2
\end{align}$$


---




## AI Statement

I've not used AI for any of the body text or code writing pertaining to actual A/B testing, but at times have used it to 
- chock up some quick website designs
- fact checker/checking for inconcsistencies
- got it to help me with animating the confusion matrix in [Parti I: Errors and Power](#errors-and-power)


# Footnotes



<!-- The Message (book) -->